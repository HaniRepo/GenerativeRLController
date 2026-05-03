#run_benchmark_suit.py
import os, json, csv
import numpy as np
from copy import deepcopy

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# your modules
from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window
from stress_wrappers import NoisyDelayedWrapper, ActionRateLimiter, SetpointJumpWrapper, ThrottleCapWrapper
from shield import SimpleThrottleShield
from conformal_shield import OneStepVTLinear, split_conformal_q, ConformalSTLShield, collect_calibration

# ----------------- config -----------------
MODEL_SRC = "ppo_f16_engine.zip"
SEED      = 42
N_EVAL    = 5   #default value is 20

# nominal env settings
BASE = dict(sp=500.0, dt=0.1, T=60.0)

# ----------------- helpers -----------------
def _unwrap(e):
    while hasattr(e, "env"):
        e = e.env
    return e

def _get_dt(env):
    if hasattr(env, "dt"): return env.dt
    if hasattr(env, "env") and hasattr(env.env, "dt"): return env.env.dt
    return env.unwrapped.dt

def make_nominal_env(sp=BASE["sp"], dt=BASE["dt"], T=BASE["T"], seed=SEED):
    return Monitor(F16EngineEnv(sp=sp, dt=dt, ep_len_s=T, seed=seed))

def switch_to_morelli(env):
    base = _unwrap(env)
    if hasattr(base, "sim") and hasattr(base.sim, "cfg") and hasattr(base.sim.cfg, "model_name"):
        base.sim.cfg.model_name = "morelli"
    elif hasattr(base, "sim") and hasattr(base.sim, "f16_model"):
        base.sim.f16_model = "morelli"

def rollout(env, model, filt=None, seed=SEED):
    obs, info = env.reset(seed=seed)
    t, vt = [], []
    base = _unwrap(env)
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        u = float(a[0])
        if filt is not None:
            u = filt(u, base)
        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        t.append(info["t"]); vt.append(info["Vt"])
        if trunc: break
    sp = _unwrap(env).sp
    return np.array(t), np.array(vt), float(sp)

def eval_many(env_builder, model, tol=0.05, window_s=10.0, n_episodes=N_EVAL):
    sats, rhos = [], []
    for k in range(n_episodes):
        env = env_builder()
        dt = _get_dt(env)
        # PPO
        t, vt, sp = rollout(env, model)
        sat, rho  = settling_spec_last_window(vt, sp=sp, dt=dt, window_s=window_s, tol=tol)
        sats.append(float(sat)); rhos.append(float(rho))
    return float(np.mean(sats)), float(np.mean(rhos))

def eval_many_stl(env_builder, model, tol=0.05, window_s=10.0, n_episodes=N_EVAL):
    sats, rhos = [], []
    for k in range(n_episodes):
        env = env_builder()
        dt = _get_dt(env)
        sh = SimpleThrottleShield(slew=0.05); sh.reset(u0=0.5)
        t, vt, sp = rollout(env, model, filt=lambda u_rl, base: sh.filter(base.sim.Vt, base.sp, u_rl))
        sat, rho  = settling_spec_last_window(vt, sp=sp, dt=dt, window_s=window_s, tol=tol)
        sats.append(float(sat)); rhos.append(float(rho))
    return float(np.mean(sats)), float(np.mean(rhos))

def eval_many_conf(env_builder, model, tol=0.05, window_s=10.0, n_episodes=N_EVAL, delta=0.30, K=6, slew=0.03, calib_eps=8):
    changed = 0
    sats, rhos = [], []

    # calibrate ONCE for this scenario
    env_cal = env_builder()
    dt = _get_dt(env_cal)
    base_cal = _unwrap(env_cal)

    Vt, pw, thr = collect_calibration(base_cal, policy=model, episodes=calib_eps, random_throttle=False, seed=123)
    pred = OneStepVTLinear()
    pred.fit(Vt, pw, thr)

    predn = [pred.predict_next(vt0, pw0, u0) for vt0, pw0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])]
    q = split_conformal_q(np.asarray(predn) - Vt[1:], delta=delta)
    print(f"[CONF CAL] q={q:.4f}")

    for k in range(n_episodes):
        env = env_builder()
        sh = ConformalSTLShield(pred, q=q, K=K, dt=dt, tol=tol, slew=slew)
        sh.reset(u0=0.5)

        def filt_fn(u_rl, base):
            nonlocal changed
            u, _ = sh.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)
            if abs(u - u_rl) >= 0.05:
                changed += 1
            return u

        t, vt, sp = rollout(env, model, filt=filt_fn)

        
        sat, rho = settling_spec_last_window(vt, sp=sp, dt=dt, window_s=window_s, tol=tol)
        print(f"[CONF EP] changed={changed}, sat={sat}, rho={rho:.4f}")
        sats.append(float(sat))
        rhos.append(float(rho))

    return float(np.mean(sats)), float(np.mean(rhos))

# ----------------- scenarios -----------------
def build_scenarios():
    #S = [S[3]]  # Rate limit only
    S = []

    # S0: Nominal (5%/10s)
    S.append(dict(
        name="Nominal (500)",
        spec=dict(tol=0.05, window_s=10.0),
        env=lambda: make_nominal_env(500.0)
    ))

    # S1/S2: setpoint shifts (still nominal plant)
    S.append(dict(
        name="Setpoint 450 (nominal)",
        spec=dict(tol=0.05, window_s=10.0),
        env=lambda: make_nominal_env(450.0)
    ))
    S.append(dict(
        name="Setpoint 550 (nominal)",
        spec=dict(tol=0.05, window_s=10.0),
        env=lambda: make_nominal_env(550.0)
    ))

    # Progressive stress (each row adds one stressor)
    # S3: rate limit only
    def env_s3():
        e = make_nominal_env()
        return ActionRateLimiter(e, slew=0.02)
    S.append(dict(name="Rate limit (slew=0.02)", spec=dict(tol=0.05, window_s=10.0), env=env_s3))

    # S4: + noise
    def env_s4():
        e = make_nominal_env()
        e = ActionRateLimiter(e, slew=0.02)
        return NoisyDelayedWrapper(e, obs_sigma=6.0, act_delay_steps=0)
    S.append(dict(name="Rate + noise (std=6)", spec=dict(tol=0.05, window_s=10.0), env=env_s4))

    # S5: + delay
    def env_s5():
        e = make_nominal_env()
        e = ActionRateLimiter(e, slew=0.02)
        return NoisyDelayedWrapper(e, obs_sigma=6.0, act_delay_steps=2)
    S.append(dict(name="Rate + noise + delay (2)", spec=dict(tol=0.05, window_s=10.0), env=env_s5))

    # S6: + throttle cap
    def env_s6():
        e = make_nominal_env()
        e = ThrottleCapWrapper(e, u_max=0.85)
        e = ActionRateLimiter(e, slew=0.02)
        return NoisyDelayedWrapper(e, obs_sigma=6.0, act_delay_steps=2)
    S.append(dict(name="Rate + noise + delay + cap (0.85)", spec=dict(tol=0.05, window_s=10.0), env=env_s6))

    # S7: + model mismatch (Morelli)
    def env_s7():
        e = make_nominal_env()
        switch_to_morelli(e)
        e = ThrottleCapWrapper(e, u_max=0.85)
        e = ActionRateLimiter(e, slew=0.02)
        return NoisyDelayedWrapper(e, obs_sigma=6.0, act_delay_steps=2)
    S.append(dict(name="… + Morelli model", spec=dict(tol=0.05, window_s=10.0), env=env_s7))

    # S8: + setpoint jump (still 5%/10s)
    def env_s8():
        e = make_nominal_env()
        switch_to_morelli(e)
        e = ThrottleCapWrapper(e, u_max=0.80)
        e = ActionRateLimiter(e, slew=0.015)
        e = NoisyDelayedWrapper(e, obs_sigma=10.0, act_delay_steps=3)
        return SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=590.0)
    S.append(dict(name="… + jump to 590 @20s", spec=dict(tol=0.05, window_s=10.0), env=env_s8))

    # S9 (final harsh row): tighten STL to 2%/5s
    def env_s9():
        e = make_nominal_env()
        switch_to_morelli(e)
        e = ThrottleCapWrapper(e, u_max=0.80)
        e = ActionRateLimiter(e, slew=0.008)
        e = NoisyDelayedWrapper(e, obs_sigma=12.0, act_delay_steps=4)
        return SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=590.0)
    S.append(dict(name="Full stress (cap+slew+noise+delay+Morelli+jump)", spec=dict(tol=0.02, window_s=5.0), env=env_s9))

    return S

# ----------------- main -----------------
def main():
    if not os.path.exists(MODEL_SRC):
        raise FileNotFoundError(f"Missing model: {MODEL_SRC}")
    # nominal env just to load model
    load_env = make_nominal_env()
    model = PPO.load(MODEL_SRC, env=load_env)

    scenarios = build_scenarios()

    rows = []
    print("\n=== Progressive Benchmark Suite ===")
    print(f"{'Scenario':48s} | {'Method':14s} | {'Sat%':>6s} | {'Mean ρ':>8s}")
    print("-"*86)

    for sc in scenarios:
        tol = sc["spec"]["tol"]; win = sc["spec"]["window_s"]

        # builders so each run starts from a fresh env for that scenario
        builder = sc["env"]

        sat_b, rho_b = eval_many(builder,      model, tol=tol, window_s=win, n_episodes=N_EVAL)
        sat_s, rho_s = eval_many_stl(builder,  model, tol=tol, window_s=win, n_episodes=N_EVAL)
        sat_c, rho_c = eval_many_conf(builder, model, tol=tol, window_s=win, n_episodes=N_EVAL,
                                      delta=0.30, K=6, slew=0.03, calib_eps=8)

        for method, sat, rho in [("PPO", sat_b, rho_b), ("PPO+STL", sat_s, rho_s), ("PPO+CONF", sat_c, rho_c)]:
            rows.append(dict(
                scenario=sc["name"], tol=tol, window_s=win,
                method=method, sat=float(sat), rho=float(rho)
            ))
            print(f"{sc['name'][:48]:48s} | {method:14s} | {sat*100:6.1f} | {rho:8.3f}")

    # write CSV/JSON
    with open("suite_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["scenario","tol","window_s","method","sat","rho"])
        w.writeheader()
        w.writerows(rows)

    with open("suite_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    #with open("suite_table.tex", "w", encoding="utf-8") as f:
    #    f.write("\n".join(lines))

    # make LaTeX table
    scenarios_order = [s["name"] for s in scenarios]
    grouped = {name: [] for name in scenarios_order}
    for r in rows:
        grouped[r["scenario"]].append(r)

    lines = []
    lines.append("\\begin{table}[t]\\centering")
    lines.append("\\caption{Progressive stress suite: STL satisfaction (\\%) and mean robustness $\\rho$.}")
    lines.append("\\label{tab:suite}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Scenario & PPO & PPO+STL & PPO+CONF\\\\")
    lines.append("\\midrule")
    for name in scenarios_order:
        trio = {r["method"]: r for r in grouped[name]}
        def fmt(m):
            sat = 100*trio[m]["sat"]; rho = trio[m]["rho"]
            return f"{sat:.0f}\\,(\\,\\,{rho:.3f}\\,)"
        lines.append(f"{name} & {fmt('PPO')} & {fmt('PPO+STL')} & {fmt('PPO+CONF')}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open("suite_table.tex","w") as f:
        f.write("\n".join(lines))

    print("\nWrote:")
    print("  - suite_results.csv")
    print("  - suite_results.json")
    print("  - suite_table.tex  (paste into LaTeX)")

if __name__ == "__main__":
    import numpy as np
    main()

