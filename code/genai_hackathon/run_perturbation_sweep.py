# genai_hackathon/run_perturbation_sweep.py

import os
import sys
import csv
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window
from stress_wrappers import (
    NoisyDelayedWrapper,
    SetpointJumpWrapper,
    ThrottleCapWrapper,
)
from conformal_shield import (
    OneStepVTLinear,
    split_conformal_q,
    ConformalSTLShield,
    collect_calibration,
)
from genai_hackathon.genai_shield import GenerativeConformalShield


OUT = "genai_hackathon/perturbation_sweep_out"
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"
SEED = 42
N = 5

AMPS = [0.00, 0.04, 0.08, 0.12, 0.16]


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def _dt(env):
    return _base_env(env).dt


def perturb_action(u_rl, step_idx, amp):
    return float(np.clip(u_rl + amp * np.sin(0.2 * step_idx), 0.0, 1.0))


def make_nominal(sp=500.0):
    return Monitor(F16EngineEnv(sp=sp, dt=0.1, ep_len_s=60.0, seed=SEED))


def make_decision_stress():
    e = make_nominal()
    e = ThrottleCapWrapper(e, u_max=0.75)
    e = NoisyDelayedWrapper(e, obs_sigma=1.0, act_delay_steps=1)
    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=520.0)
    return e


def build_predictor(env, model, calib_eps=3, delta=0.30):
    base = _base_env(env)

    Vt, pw, thr = collect_calibration(
        base,
        policy=model,
        episodes=calib_eps,
        random_throttle=False,
        seed=123,
    )

    pred = OneStepVTLinear()
    pred.fit(Vt, pw, thr)

    pred_next = [
        pred.predict_next(v0, p0, u0)
        for v0, p0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])
    ]

    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=delta)
    return pred, q


def rollout_ppo(env, model, seed, tol, window_s, amp):
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    vt = []
    done = False

    while not done:
        a, _ = model.predict(obs, deterministic=True)
        u_raw = float(np.asarray(a).reshape(-1)[0])
        u = perturb_action(u_raw, len(vt), amp)

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        vt.append(info["Vt"])

        if trunc:
            break

    sat, rho = settling_spec_last_window(
        np.asarray(vt),
        sp=base.sp,
        dt=_dt(env),
        window_s=window_s,
        tol=tol,
    )

    return float(sat), float(rho)


def rollout_conf(env, model, seed, tol, window_s, amp):
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    pred, q = build_predictor(env, model, calib_eps=3, delta=0.30)

    shield = ConformalSTLShield(
        pred,
        q=q,
        K=4,
        dt=_dt(env),
        tol=tol,
        slew=0.03,
    )
    # initialize shield with the first perturbed PPO action
    a0, _ = model.predict(obs, deterministic=True)
    u0_raw = float(np.asarray(a0).reshape(-1)[0])
    u0 = perturb_action(u0_raw, 0, amp)
    shield.reset(u0=u0)
    shield.debug = False

    vt = []
    done = False

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_raw = float(np.asarray(a_rl).reshape(-1)[0])
        u_rl = perturb_action(u_raw, len(vt), amp)

        u, _ = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        vt.append(info["Vt"])

        if trunc:
            break

    sat, rho = settling_spec_last_window(
        np.asarray(vt),
        sp=base.sp,
        dt=_dt(env),
        window_s=window_s,
        tol=tol,
    )

    return float(sat), float(rho)


def rollout_genai(env, model, seed, tol, window_s, amp):
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    pred, q = build_predictor(env, model, calib_eps=3, delta=0.30)

    shield = GenerativeConformalShield(
        pred,
        q=q,
        K=8,
        dt=_dt(env),
        tol=tol,
        slew=0.08,
        candidate_offsets=(-0.12, -0.06, 0.0, 0.06, 0.12),
    )
    # initialize shield with the first perturbed PPO action
    a0, _ = model.predict(obs, deterministic=True)
    u0_raw = float(np.asarray(a0).reshape(-1)[0])
    u0 = perturb_action(u0_raw, 0, amp)
    shield.reset(u0=u0)
    shield.debug = False

    vt = []
    done = False

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_raw = float(np.asarray(a_rl).reshape(-1)[0])
        u_rl = perturb_action(u_raw, len(vt), amp)

        u, _ = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        vt.append(info["Vt"])

        if trunc:
            break

    sat, rho = settling_spec_last_window(
        np.asarray(vt),
        sp=base.sp,
        dt=_dt(env),
        window_s=window_s,
        tol=tol,
    )

    return float(sat), float(rho)


def eval_method(method, model, amp, n=N):
    sats = []
    rhos = []

    for i in range(n):
        env = make_decision_stress()
        seed = SEED + i

        if method == "PPO":
            sat, rho = rollout_ppo(env, model, seed, tol=0.03, window_s=8.0, amp=amp)
        elif method == "PPO+CONF":
            sat, rho = rollout_conf(env, model, seed, tol=0.03, window_s=8.0, amp=amp)
        elif method == "PPO+GENAI":
            sat, rho = rollout_genai(env, model, seed, tol=0.03, window_s=8.0, amp=amp)
        else:
            raise ValueError(f"Unknown method: {method}")

        sats.append(sat)
        rhos.append(rho)

    return float(np.mean(sats)), float(np.mean(rhos))


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

    rows = []

    print("\n=== Perturbation Sweep: DecisionStress ===")
    print(f"{'Amp':>5s} | {'Method':10s} | {'Sat%':>6s} | {'Mean rho':>9s}")
    print("-" * 45)

    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]

    for amp in AMPS:
        for method in methods:
            sat, rho = eval_method(method, model, amp, n=N)

            print(f"{amp:5.2f} | {method:10s} | {100*sat:6.1f} | {rho:9.4f}")

            rows.append(
                {
                    "amp": amp,
                    "method": method,
                    "sat": sat,
                    "rho": rho,
                    "sat_percent": 100 * sat,
                }
            )

    csv_path = os.path.join(OUT, "perturbation_sweep.csv")
    json_path = os.path.join(OUT, "perturbation_sweep.json")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["amp", "method", "sat", "sat_percent", "rho"],
        )
        w.writeheader()
        w.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("\nSaved:")
    print(f" - {csv_path}")
    print(f" - {json_path}")


if __name__ == "__main__":
    main()