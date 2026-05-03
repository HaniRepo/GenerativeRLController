# genai_hackathon/run_speed_envelope_dual_suite.py

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
from stress_wrappers import SetpointJumpWrapper

from conformal_shield import (
    OneStepVTLinear,
    split_conformal_q,
    ConformalSTLShield,
    collect_calibration,
)
from genai_hackathon.genai_shield import GenerativeConformalShield


OUT = "genai_hackathon/speed_envelope_dual_suite_out"
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"
SEED = 42
N = 10


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def _dt(env):
    return _base_env(env).dt


def make_nominal(sp=500.0):
    return Monitor(F16EngineEnv(sp=sp, dt=0.1, ep_len_s=60.0, seed=SEED))


def make_speed_envelope_stress():
    e = make_nominal(sp=500.0)
    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=560.0)
    return e


def speed_envelope_spec(vt_trace, sp_trace, tol_under=0.05, tol_over=0.03, start_frac=0.0):
    vt = np.asarray(vt_trace, dtype=float)
    sp = np.asarray(sp_trace, dtype=float)

    start = int(start_frac * len(vt))
    vt = vt[start:]
    sp = sp[start:]

    upper = (sp * (1.0 + tol_over) - vt) / np.maximum(sp, 1e-6)
    lower = (vt - sp * (1.0 - tol_under)) / np.maximum(sp, 1e-6)

    rho = float(min(np.min(upper), np.min(lower)))
    return rho >= 0.0, rho


def action_metrics(actions):
    actions = np.asarray(actions, dtype=float)
    if len(actions) <= 1:
        return 0.0, 0.0, float(np.mean(actions)) if len(actions) else 0.0
    du = np.abs(np.diff(actions))
    return float(np.mean(du)), float(np.max(du)), float(np.mean(actions))


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


def rollout(method, env, model, seed):
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    vt, sp_hist, actions = [], [], []
    changed = 0
    bad = 0

    shield = None

    if method == "PPO+CONF":
        pred, q = build_predictor(env, model)
        shield = ConformalSTLShield(
            pred,
            q=q,
            K=10,
            dt=_dt(env),
            tol=0.03,
            slew=0.03,
        )
        a0, _ = model.predict(obs, deterministic=True)
        u0 = float(np.asarray(a0).reshape(-1)[0])
        shield.reset(u0=u0)
        shield.debug = False

    elif method == "PPO+GENAI":
        pred, q = build_predictor(env, model)
        shield = GenerativeConformalShield(
            pred,
            q=q,
            K=10,
            dt=_dt(env),
            tol=0.03,
            slew=0.25,
            candidate_offsets=(-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30),
        )
        a0, _ = model.predict(obs, deterministic=True)
        u0 = float(np.asarray(a0).reshape(-1)[0])
        shield.reset(u0=u0)
        shield.debug = False

    done = False

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_rl = float(np.asarray(a_rl).reshape(-1)[0])

        if method == "PPO":
            u = u_rl
            pred_rho = 999.0
        elif method == "PPO+CONF":
            u, pred_rho = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)
        elif method == "PPO+GENAI":
            u, pred_rho = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)
        else:
            raise ValueError(method)

        if method == "PPO+GENAI":
            if abs(u - u_rl) >= 0.03:
                changed += 1
            if pred_rho < 0.0:
                bad += 1

        actions.append(u)

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))

        vt.append(info["Vt"])
        sp_hist.append(info["sp"])

        if trunc:
            break

    mean_du, max_du, mean_u = action_metrics(actions)

    return {
        "vt": np.asarray(vt),
        "sp": np.asarray(sp_hist),
        "mean_du": mean_du,
        "max_du": max_du,
        "mean_u": mean_u,
        "changed": changed,
        "bad": bad,
    }


def evaluate_method(method, model, start_frac, n=N):
    sats, rhos = [], []
    mean_dus, max_dus, mean_us = [], [], []
    changed_all, bad_all = [], []

    for i in range(n):
        env = make_speed_envelope_stress()
        res = rollout(method, env, model, SEED + i)

        sat, rho = speed_envelope_spec(
            res["vt"],
            res["sp"],
            tol_under=0.05,
            tol_over=0.03,
            start_frac=start_frac,
        )

        sats.append(float(sat))
        rhos.append(float(rho))
        mean_dus.append(res["mean_du"])
        max_dus.append(res["max_du"])
        mean_us.append(res["mean_u"])
        changed_all.append(res["changed"])
        bad_all.append(res["bad"])

    return {
        "sat": float(np.mean(sats)),
        "rho": float(np.mean(rhos)),
        "mean_du": float(np.mean(mean_dus)),
        "max_du": float(np.mean(max_dus)),
        "mean_u": float(np.mean(mean_us)),
        "changed": float(np.mean(changed_all)),
        "bad": float(np.mean(bad_all)),
    }


def run_table(model, table_name, start_frac):
    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]
    rows = []

    print(f"\n=== {table_name} ===")
    print(
        f"{'Method':10s} | {'Sat%':>6s} | {'Mean rho':>9s} | "
        f"{'mean u':>7s} | {'changed':>7s}"
    )
    print("-" * 60)

    for method in methods:
        res = evaluate_method(method, model, start_frac=start_frac, n=N)

        print(
            f"{method:10s} | {100*res['sat']:6.1f} | "
            f"{res['rho']:9.4f} | {res['mean_u']:7.4f} | {res['changed']:7.1f}"
        )

        rows.append(
            {
                "table": table_name,
                "start_frac": start_frac,
                "method": method,
                "sat": res["sat"],
                "sat_percent": 100 * res["sat"],
                "rho": res["rho"],
                "mean_du": res["mean_du"],
                "max_du": res["max_du"],
                "mean_u": res["mean_u"],
                "changed": res["changed"],
                "bad": res["bad"],
            }
        )

    return rows


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

    all_rows = []

    all_rows += run_table(
        model,
        table_name="Speed Envelope — Strict Full Trace",
        start_frac=0.0,
    )

    all_rows += run_table(
        model,
        table_name="Speed Envelope — Post-Transient",
        start_frac=0.35,
    )

    csv_path = os.path.join(OUT, "speed_envelope_dual_results.csv")
    json_path = os.path.join(OUT, "speed_envelope_dual_results.json")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "table",
                "start_frac",
                "method",
                "sat",
                "sat_percent",
                "rho",
                "mean_du",
                "max_du",
                "mean_u",
                "changed",
                "bad",
            ],
        )
        w.writeheader()
        w.writerows(all_rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    print("\nSaved:")
    print(f" - {csv_path}")
    print(f" - {json_path}")


if __name__ == "__main__":
    main()