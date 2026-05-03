# run_mini_suite.py
import os
import csv
import json
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window
from stress_wrappers import (
    ActionRateLimiter,
    NoisyDelayedWrapper,
    SetpointJumpWrapper,
    ThrottleCapWrapper,
)

from shield import SimpleThrottleShield
from conformal_shield import (
    OneStepVTLinear,
    split_conformal_q,
    ConformalSTLShield,
    collect_calibration,
)

OUT = "mini_suite_out"
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"
SEED = 42
N = 5  # quick run for now; increase later for paper


# ---------- utils ----------
def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def _dt(env):
    return _base_env(env).dt


def rollout(env, model, seed=None):
    obs, info = env.reset(seed=seed)
    vt = []
    done = False

    while not done:
        a, _ = model.predict(obs, deterministic=True)
        u = float(np.asarray(a).reshape(-1)[0])

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        vt.append(info["Vt"])

        if trunc:
            break

    return np.asarray(vt), info["sp"]


def eval_many(env, model, tol, window_s, n=N):
    sats = []
    rhos = []

    for i in range(n):
        vt, sp = rollout(env, model, seed=SEED + i)
        sat, rho = settling_spec_last_window(
            vt, sp=sp, dt=_dt(env), window_s=window_s, tol=tol
        )
        sats.append(float(sat))
        rhos.append(float(rho))

    return float(np.mean(sats)), float(np.mean(rhos))


def eval_stl(env, model, tol, window_s, n=N):
    sats = []
    rhos = []

    for i in range(n):
        obs, info = env.reset(seed=SEED + i)
        base = _base_env(env)

        sh = SimpleThrottleShield(slew=0.05)
        sh.reset(u0=0.5)

        vt = []
        done = False

        while not done:
            a_rl, _ = model.predict(obs, deterministic=True)
            u_rl = float(np.asarray(a_rl).reshape(-1)[0])

            u = sh.filter(base.sim.Vt, base.sp, u_rl)

            obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
            vt.append(info["Vt"])

            if trunc:
                break

        sat, rho = settling_spec_last_window(
            np.asarray(vt), sp=base.sp, dt=_dt(env), window_s=window_s, tol=tol
        )
        sats.append(float(sat))
        rhos.append(float(rho))

    return float(np.mean(sats)), float(np.mean(rhos))

#slew = 0.03 was too aggressive

def eval_conf (env, model, tol, window_s, n=N, K=4, delta=0.30, slew=0.08, calib_eps=3 ):
    sats = []
    rhos = []
    

    base0 = _base_env(env)

    Vt, pw, thr = collect_calibration(
        base0, policy=model, episodes=calib_eps, random_throttle=False, seed=123
    )

    pred = OneStepVTLinear()
    pred.fit(Vt, pw, thr)

    pred_next = [
        pred.predict_next(v0, p0, u0)
        for v0, p0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])
    ]
    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=delta)

    for i in range(n):
        n_changed = 0
        n_risky = 0
        obs, info = env.reset(seed=SEED + i)
        base = _base_env(env)

        sh = ConformalSTLShield(pred, q=q, K=K, dt=_dt(env), tol=tol, slew=slew)
        a0, _ = model.predict(obs, deterministic=True)
        u0 = float(np.asarray(a0).reshape(-1)[0])
        sh.reset(u0=u0)

        vt = []
        done = False

        while not done:
            a_rl, _ = model.predict(obs, deterministic=True)
            u_rl = float(np.asarray(a_rl).reshape(-1)[0])

            u, rho = sh.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)
            if abs(u - u_rl) >= 0.05:
                n_changed += 1
            if rho < 0.01:
                n_risky += 1

            obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
            vt.append(info["Vt"])

            if trunc:
                break
        print(f"[CONF EP] changed={n_changed}, risky={n_risky}")        
        sat, rho = settling_spec_last_window(
            np.asarray(vt), sp=base.sp, dt=_dt(env), window_s=window_s, tol=tol
        )
        sats.append(float(sat))
        rhos.append(float(rho))
    
    return float(np.mean(sats)), float(np.mean(rhos))


# ---------- scenarios ----------
def make_nominal(sp=500.0):
    return Monitor(F16EngineEnv(sp=sp, dt=0.1, ep_len_s=60.0, seed=SEED))


def make_s1():
    e = make_nominal()
    e = ActionRateLimiter(e, slew=0.015)
    e = NoisyDelayedWrapper(e, obs_sigma=6.0, act_delay_steps=1)
    return e


def make_s2():
    e = make_s1()
    e = ThrottleCapWrapper(e, u_max=0.81)
    e = NoisyDelayedWrapper(e, obs_sigma=6.0, act_delay_steps=4)
    return e


def make_s3():
    e = make_s2()

    b = _base_env(e)
    if hasattr(b, "sim") and hasattr(b.sim, "cfg") and hasattr(b.sim.cfg, "model_name"):
        b.sim.cfg.model_name = "morelli"
    elif hasattr(b, "sim") and hasattr(b.sim, "f16_model"):
        b.sim.f16_model = "morelli"

    e = SetpointJumpWrapper(e, t_jump_s=22.0, sp_new=595.0)
    return e


SUITE = [
    ("S0 Nominal (5%/10s)", make_nominal, dict(tol=0.05, window_s=10.0)),
    ("S1 Mild (4%/8s)", make_s1, dict(tol=0.04, window_s=8.0)),
    ("S2 Moderate (3%/6s)", make_s2, dict(tol=0.03, window_s=6.0)),
    ("S3 Strong (2%/5s)", make_s3, dict(tol=0.02, window_s=5.0)),
]


def main():
    os.makedirs(OUT, exist_ok=True)

    env0 = make_nominal()
    model = PPO.load(
        MODEL,
        env=env0,
        custom_objects={
            "learning_rate": 3e-4,
            "clip_range": 0.2,
        },
    )

    header = ["scenario", "method", "tol", "window_s", "sat", "rho"]
    rows = []

    print("\n=== Mini Benchmark (4 steps) ===")
    print(f"{'Scenario':28s} | {'Method':10s} | {'Sat%':>6s} | {'Mean ρ':>7s}")
    print("-" * 64)

    methods = [
        ("PPO", eval_many),
        ("PPO+STL", eval_stl),
        ("PPO+CONF", eval_conf),
    ]

    for name, maker, spec in SUITE:
        env = maker()

        for method, fn in methods:
            sat, rho = fn(env, model, **spec)
            print(f"{name:28s} | {method:10s} | {100*sat:6.1f} | {rho:7.3f}")
            rows.append(
                dict(
                    scenario=name,
                    method=method,
                    tol=spec["tol"],
                    window_s=spec["window_s"],
                    sat=sat,
                    rho=rho,
                )
            )

    with open(os.path.join(OUT, "mini_suite.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(OUT, "mini_suite.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print("\nSaved:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()