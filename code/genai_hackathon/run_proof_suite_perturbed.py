# genai_hackathon/run_proof_suite.py

import os
import sys
import csv
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

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


OUT = "genai_hackathon/proof_suite_perturbed_out"
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"
SEED = 42
N = 5


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def _dt(env):
    return _base_env(env).dt


def action_metrics(actions):
    actions = np.asarray(actions, dtype=float)
    if len(actions) <= 1:
        return 0.0, 0.0, float(np.mean(actions)) if len(actions) else 0.0

    du = np.abs(np.diff(actions))
    return float(np.mean(du)), float(np.max(du)), float(np.mean(actions))

def perturb_action(u_rl, step_idx, amp=0.08):
    return float(np.clip(u_rl + amp * np.sin(0.2 * step_idx), 0.0, 1.0))

def make_nominal(sp=500.0):
    return Monitor(F16EngineEnv(sp=sp, dt=0.1, ep_len_s=60.0, seed=SEED))


def make_decision_stress():
    e = make_nominal()
    e = ThrottleCapWrapper(e, u_max=0.75)
    e = NoisyDelayedWrapper(e, obs_sigma=1.0, act_delay_steps=1)
    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=520.0)
    return e


SCENARIOS = [
    ("Nominal", make_nominal, dict(tol=0.05, window_s=10.0)),
    ("DecisionStress", make_decision_stress, dict(tol=0.03, window_s=8.0)),
]


def rollout_ppo(env, model, seed, tol, window_s):
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    vt, actions = [], []
    done = False

    while not done:
        a, _ = model.predict(obs, deterministic=True)
        u_raw = float(np.asarray(a).reshape(-1)[0])
        u = perturb_action(u_raw, len(vt), amp=0.08)
        actions.append(u)

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        vt.append(info["Vt"])

        if trunc:
            break

    sat, rho = settling_spec_last_window(
        np.asarray(vt), sp=base.sp, dt=_dt(env), window_s=window_s, tol=tol
    )

    mean_du, max_du, mean_u = action_metrics(actions)

    return dict(vt=np.asarray(vt), sat=sat, rho=rho,
                mean_du=mean_du, max_du=max_du, mean_u=mean_u)


def build_predictor(env, model, calib_eps=3, delta=0.30):
    base = _base_env(env)

    Vt, pw, thr = collect_calibration(
        base, policy=model, episodes=calib_eps, random_throttle=False, seed=123
    )

    pred = OneStepVTLinear()
    pred.fit(Vt, pw, thr)

    pred_next = [
        pred.predict_next(v0, p0, u0)
        for v0, p0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])
    ]

    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=delta)

    return pred, q


def rollout_conf(env, model, seed, tol, window_s):
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    pred, q = build_predictor(env, model)

    shield = ConformalSTLShield(pred, q=q, K=4, dt=_dt(env), tol=tol, slew=0.03)
    shield.reset(u0=0.5)

    vt, actions = [], []
    done = False

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_raw = float(np.asarray(a_rl).reshape(-1)[0])
        u_rl = perturb_action(u_raw, len(vt), amp=0.08)


        u, _ = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)
        actions.append(u)

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        vt.append(info["Vt"])

        if trunc:
            break

    sat, rho = settling_spec_last_window(
        np.asarray(vt), sp=base.sp, dt=_dt(env), window_s=window_s, tol=tol
    )

    mean_du, max_du, mean_u = action_metrics(actions)

    return dict(vt=np.asarray(vt), sat=sat, rho=rho,
                mean_du=mean_du, max_du=max_du, mean_u=mean_u)


def rollout_genai(env, model, seed, tol, window_s):
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    # ✅ back to stable predictor
    pred, q = build_predictor(env, model)

    shield = GenerativeConformalShield(
        pred,
        q=q,
        K=8,
        dt=_dt(env),
        tol=tol,
        slew=0.08,
        candidate_offsets=(-0.12, -0.06, 0.0, 0.06, 0.12),
    )
    shield.reset(u0=0.5)

    vt, actions = [], []
    changed, bad = 0, 0
    done = False

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_raw = float(np.asarray(a_rl).reshape(-1)[0])
        u_rl = perturb_action(u_raw, len(vt), amp=0.08)

        u, rho = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)
        actions.append(u)

        if abs(u - u_rl) >= 0.03:
            changed += 1
        if rho < 0.0:
            bad += 1

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        vt.append(info["Vt"])

        if trunc:
            break

    sat, rho = settling_spec_last_window(
        np.asarray(vt), sp=base.sp, dt=_dt(env), window_s=window_s, tol=tol
    )

    mean_du, max_du, mean_u = action_metrics(actions)

    print(f"[GENAI EP] changed={changed}, bad={bad}, sat={sat}, rho={rho:.4f}")

    return dict(vt=np.asarray(vt), sat=sat, rho=rho,
                mean_du=mean_du, max_du=max_du, mean_u=mean_u)


def evaluate_method(env_maker, model, method_name, tol, window_s):
    sats, rhos, dus = [], [], []

    for i in range(N):
        env = env_maker()

        if method_name == "PPO":
            res = rollout_ppo(env, model, SEED+i, tol, window_s)
        elif method_name == "PPO+CONF":
            res = rollout_conf(env, model, SEED+i, tol, window_s)
        else:
            res = rollout_genai(env, model, SEED+i, tol, window_s)

        sats.append(res["sat"])
        rhos.append(res["rho"])
        dus.append(res["mean_du"])

    return np.mean(sats), np.mean(rhos), np.mean(dus)


def main():
    env0 = make_nominal()
    model = PPO.load(MODEL, env=env0)

    print("\n=== Perturbed PPO Proof-of-Concept Suite ===")

    for name, maker, spec in SCENARIOS:
        for method in ["PPO", "PPO+CONF", "PPO+GENAI"]:
            sat, rho, du = evaluate_method(
                maker, model, method,
                tol=spec["tol"], window_s=spec["window_s"]
            )

            print(f"{name:14s} | {method:10s} | {100*sat:6.1f} | {rho:7.4f}")


if __name__ == "__main__":
    main()