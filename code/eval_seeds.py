#eval_seeds.py

import json, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from f16_engine_env import F16EngineEnv
from stress_wrappers import NoisyDelayedWrapper, ActionRateLimiter, SetpointJumpWrapper, ThrottleCapWrapper
from stl_monitor import settling_spec_last_window

N_EVAL = 20
SEEDS = [0, 1, 2, 3, 4]

def _get_dt(env):
    if hasattr(env,'dt'): return env.dt
    if hasattr(env,'env') and hasattr(env.env,'dt'): return env.env.dt
    return env.unwrapped.dt

def _base_env(e):
    while hasattr(e,'env'): e = e.env
    return e

def rollout(env, model):
    obs, info = env.reset()
    t_hist, vt_hist, sp = [], [], info.get("sp", _base_env(env).sp)
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(act)
        t_hist.append(info["t"]); vt_hist.append(info["Vt"])
        if trunc: break
    return np.array(t_hist), np.array(vt_hist), sp

def eval_one_env(env, model, tol=0.02, window_s=5.0, n_episodes=20):
    sats, rhos = [], []
    dt = _get_dt(env)
    for _ in range(n_episodes):
        t, vt, sp = rollout(env, model)
        sat, rho = settling_spec_last_window(vt, sp=sp, dt=dt, window_s=window_s, tol=tol)
        sats.append(float(sat)); rhos.append(float(rho))
    return float(np.mean(sats)), float(np.mean(rhos))

def make_stress_env(sp=500.0, dt=0.1, horizon=60.0, seed=0):
    env = F16EngineEnv(sp=sp, dt=dt, ep_len_s=horizon, seed=seed)
    env = Monitor(env)
    base = _base_env(env)
    # flip to Morelli
    if hasattr(base, "sim") and hasattr(base.sim, "cfg") and hasattr(base.sim.cfg, "model_name"):
        base.sim.cfg.model_name = "morelli"
    elif hasattr(base, "sim") and hasattr(base.sim, "f16_model"):
        base.sim.f16_model = "morelli"
    # stress stack
    env = ThrottleCapWrapper(env, u_max=0.80)
    env = ActionRateLimiter(env, slew=0.008)
    env = NoisyDelayedWrapper(env, obs_sigma=12.0, act_delay_steps=4)
    env = SetpointJumpWrapper(env, t_jump_s=20.0, sp_new=590.0)
    return env

def main():
    res = {"seeds": [], "stress": {"tol":0.02, "window_s":5.0}}
    for s in SEEDS:
        env = make_stress_env(seed=s)
        model = PPO.load("ppo_f16_engine.zip", env=env)
        sat_base, rho_base = eval_one_env(env, model, tol=0.02, window_s=5.0, n_episodes=N_EVAL)

        # STL shield
        from shield import SimpleThrottleShield
        def eval_shielded(env, model):
            dt = _get_dt(env)
            sats, rhos = [], []
            for _ in range(N_EVAL):
                obs, info = env.reset()
                base = _base_env(env)
                sh = SimpleThrottleShield(slew=0.05); sh.reset(u0=0.5)
                vt_hist=[]; done=False
                while not done:
                    act,_ = model.predict(obs, deterministic=True)
                    u = sh.filter(base.sim.Vt, base.sp, float(act[0]))
                    obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
                    vt_hist.append(info["Vt"])
                    if trunc: break
                sat, rho = settling_spec_last_window(np.array(vt_hist), sp=base.sp, dt=dt, window_s=5.0, tol=0.02)
                sats.append(float(sat)); rhos.append(float(rho))
            return float(np.mean(sats)), float(np.mean(rhos))
        sat_sh, rho_sh = eval_shielded(env, model)

        # Conformal shield
        from conformal_shield import OneStepVTLinear, split_conformal_q, ConformalSTLShield, collect_calibration
        base = _base_env(env)
        Vt, pw, thr = collect_calibration(base, policy=model, episodes=8, random_throttle=False, seed=123+s)
        pred = OneStepVTLinear(); pred.fit(Vt, pw, thr)
        pred_next = [pred.predict_next(vt0,pw0,u0) for vt0,pw0,u0 in zip(Vt[:-1], pw[:-1], thr[:-1])]
        q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=0.30)
        def eval_conf(env, model):
            dt = _get_dt(env)
            sats, rhos = [], []
            for _ in range(N_EVAL):
                obs, info = env.reset()
                base2 = _base_env(env)
                #sh = ConformalSTLShield(pred, q=q, K=6, dt=dt, tol=0.05, slew=0.03)
                sh = ConformalSTLShield(pred, q=q, K=6, dt=dt, tol=0.02, slew=0.03)
                sh.reset(u0=0.5)
                vt_hist=[]; done=False
                while not done:
                    act,_ = model.predict(obs, deterministic=True)
                    u,_ = sh.filter(base2.sim.Vt, base2.sim.pow, base2.sp, float(act[0]))
                    obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
                    vt_hist.append(info["Vt"])
                    if trunc: break
                sat, rho = settling_spec_last_window(np.array(vt_hist), sp=base2.sp, dt=dt, window_s=5.0, tol=0.02)
                sats.append(float(sat)); rhos.append(float(rho))
            return float(np.mean(sats)), float(np.mean(rhos))
        sat_conf, rho_conf = eval_conf(env, model)

        res["seeds"].append({
            "seed": s,
            "baseline": {"sat": sat_base, "rho": rho_base},
            "shield":   {"sat": sat_sh,   "rho": rho_sh},
            "conformal":{"sat": sat_conf, "rho": rho_conf},
        })

    # aggregate
    for key in ["baseline","shield","conformal"]:
        sats = [d[key]["sat"] for d in res["seeds"]]
        rhos = [d[key]["rho"] for d in res["seeds"]]
        res[key+"_agg"] = {
            "sat_mean": float(np.mean(sats)), "sat_std": float(np.std(sats)),
            "rho_mean": float(np.mean(rhos)), "rho_std": float(np.std(rhos)),
        }
    print(json.dumps(res, indent=2))
    with open("shield_pack/seed_eval.json","w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
