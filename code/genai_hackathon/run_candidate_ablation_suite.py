# genai_hackathon/run_candidate_ablation_suite.py

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


OUT = "genai_hackathon/candidate_ablation_suite_out"
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"
SEED = 42
N = 10


GENAI_CONFIGS = {
    "GENAI-1": (0.0,),
    "GENAI-3": (-0.20, 0.0, 0.20),
    "GENAI-5": (-0.30, -0.15, 0.0, 0.15, 0.30),
    "GENAI-7": (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30),
     "GENAI-9": (-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40),
}


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def _dt(env):
    return _base_env(env).dt


def make_nominal(sp=500.0):
    return Monitor(F16EngineEnv(sp=sp, dt=0.1, ep_len_s=60.0, seed=SEED))


def make_overspeed_stress():
    e = make_nominal(sp=500.0)
    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=560.0)
    return e


def overspeed_tracking_spec(vt_trace, sp_trace, tol_over=0.03, tol_track=0.05, last_frac=0.25):
    vt_trace = np.asarray(vt_trace, dtype=float)
    sp_trace = np.asarray(sp_trace, dtype=float)

    over_margin = (sp_trace * (1.0 + tol_over) - vt_trace) / np.maximum(sp_trace, 1e-6)
    rho_over = float(np.min(over_margin))

    n = len(vt_trace)
    start = int((1.0 - last_frac) * n)

    vt_last = vt_trace[start:]
    sp_last = sp_trace[start:]

    track_margin = tol_track - np.abs((vt_last - sp_last) / np.maximum(sp_last, 1e-6))
    rho_track = float(np.min(track_margin))

    rho = min(rho_over, rho_track)
    return rho >= 0.0, rho


def action_metrics(actions):
    actions = np.asarray(actions, dtype=float)
    if len(actions) <= 1:
        return 0.0, 0.0, float(np.mean(actions)) if len(actions) else 0.0

    du = np.abs(np.diff(actions))
    return float(np.mean(du)), float(np.max(du)), float(np.mean(actions))


def recovery_metrics(vt_trace, sp_trace, dt=0.1, jump_time=20.0, band=0.05):
    vt = np.asarray(vt_trace, dtype=float)
    sp = np.asarray(sp_trace, dtype=float)

    if len(vt) == 0:
        return np.nan, np.nan, np.nan

    start = int(jump_time / dt)

    if start >= len(vt):
        start = int(0.75 * len(vt))

    vt_post = vt[start:]
    sp_post = sp[start:]

    if len(vt_post) == 0:
        return np.nan, np.nan, np.nan

    err = np.abs((vt_post - sp_post) / np.maximum(sp_post, 1e-6))

    rmse = float(np.sqrt(np.mean(err ** 2)))
    iae = float(np.mean(err))

    inside = err <= band
    recovery_time = len(err) * dt

    for i in range(len(inside)):
        if np.all(inside[i:]):
            recovery_time = i * dt
            break

    return rmse, iae, recovery_time


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


def rollout_ppo(env, model, seed, tol_over, tol_track):
    obs, info = env.reset(seed=seed)

    vt, sp_hist, actions = [], [], []
    done = False

    while not done:
        a, _ = model.predict(obs, deterministic=True)
        u = float(np.asarray(a).reshape(-1)[0])
        actions.append(u)

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))

        vt.append(info["Vt"])
        sp_hist.append(info["sp"])

        if trunc:
            break

    sat, rho = overspeed_tracking_spec(
        vt, sp_hist, tol_over=tol_over, tol_track=tol_track, last_frac=0.25
    )

    mean_du, max_du, mean_u = action_metrics(actions)
    rmse, iae, rec_time = recovery_metrics(vt, sp_hist)

    return {
        "sat": float(sat),
        "rho": float(rho),
        "mean_du": mean_du,
        "max_du": max_du,
        "mean_u": mean_u,
        "rmse": rmse,
        "iae": iae,
        "recovery_time": rec_time,
        "changed": 0,
        "bad": 0,
    }


def rollout_conf(env, model, seed, tol_over, tol_track):
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    pred, q = build_predictor(env, model)

    shield = ConformalSTLShield(
        pred,
        q=q,
        K=7,
        dt=_dt(env),
        tol=tol_over,
        slew=0.03,
    )

    a0, _ = model.predict(obs, deterministic=True)
    u0 = float(np.asarray(a0).reshape(-1)[0])
    shield.reset(u0=u0)
    shield.debug = False

    vt, sp_hist, actions = [], [], []
    changed, bad = 0, 0
    done = False

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_rl = float(np.asarray(a_rl).reshape(-1)[0])

        u, pred_rho = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)
        actions.append(u)

        if abs(u - u_rl) >= 0.03:
            changed += 1
        if pred_rho < 0.0:
            bad += 1

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))

        vt.append(info["Vt"])
        sp_hist.append(info["sp"])

        if trunc:
            break

    sat, rho = overspeed_tracking_spec(
        vt, sp_hist, tol_over=tol_over, tol_track=tol_track, last_frac=0.25
    )

    mean_du, max_du, mean_u = action_metrics(actions)
    rmse, iae, rec_time = recovery_metrics(vt, sp_hist)

    return {
        "sat": float(sat),
        "rho": float(rho),
        "mean_du": mean_du,
        "max_du": max_du,
        "mean_u": mean_u,
        "rmse": rmse,
        "iae": iae,
        "recovery_time": rec_time,
        "changed": changed,
        "bad": bad,
    }


def rollout_genai(env, model, seed, tol_over, tol_track, candidate_offsets, label):
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    pred, q = build_predictor(env, model)

    shield = GenerativeConformalShield(
        pred,
        q=q,
        K=10,
        dt=_dt(env),
        tol=tol_over,
        slew=0.25,
        candidate_offsets=candidate_offsets,
    )

    a0, _ = model.predict(obs, deterministic=True)
    u0 = float(np.asarray(a0).reshape(-1)[0])
    shield.reset(u0=u0)
    shield.debug = False

    vt, sp_hist, actions = [], [], []
    changed, bad = 0, 0
    done = False

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_rl = float(np.asarray(a_rl).reshape(-1)[0])

        u, pred_rho = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)
        actions.append(u)

        if abs(u - u_rl) >= 0.03:
            changed += 1
        if pred_rho < 0.0:
            bad += 1

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))

        vt.append(info["Vt"])
        sp_hist.append(info["sp"])

        if trunc:
            break

    sat, rho = overspeed_tracking_spec(
        vt, sp_hist, tol_over=tol_over, tol_track=tol_track, last_frac=0.25
    )

    mean_du, max_du, mean_u = action_metrics(actions)
    rmse, iae, rec_time = recovery_metrics(vt, sp_hist)

    print(
        f"[{label} EP] changed={changed}, bad={bad}, "
        f"sat={sat}, rho={rho:.4f}"
    )

    return {
        "sat": float(sat),
        "rho": float(rho),
        "mean_du": mean_du,
        "max_du": max_du,
        "mean_u": mean_u,
        "rmse": rmse,
        "iae": iae,
        "recovery_time": rec_time,
        "changed": changed,
        "bad": bad,
    }


def evaluate_method(method, model, tol_over, tol_track, n=N):
    sats, rhos = [], []
    mean_dus, max_dus, mean_us = [], [], []
    rmses, iaes, rec_times = [], [], []
    changeds, bads = [], []

    for i in range(n):
        env = make_overspeed_stress()
        seed = SEED + i

        if method == "PPO":
            res = rollout_ppo(env, model, seed, tol_over, tol_track)
        elif method == "PPO+CONF":
            res = rollout_conf(env, model, seed, tol_over, tol_track)
        elif method in GENAI_CONFIGS:
            res = rollout_genai(
                env,
                model,
                seed,
                tol_over,
                tol_track,
                candidate_offsets=GENAI_CONFIGS[method],
                label=method,
            )
        else:
            raise ValueError(method)

        sats.append(res["sat"])
        rhos.append(res["rho"])
        mean_dus.append(res["mean_du"])
        max_dus.append(res["max_du"])
        mean_us.append(res["mean_u"])
        rmses.append(res["rmse"])
        iaes.append(res["iae"])
        rec_times.append(res["recovery_time"])
        changeds.append(res["changed"])
        bads.append(res["bad"])

    return {
        "sat": float(np.mean(sats)),
        "rho": float(np.mean(rhos)),
        "mean_du": float(np.mean(mean_dus)),
        "max_du": float(np.mean(max_dus)),
        "mean_u": float(np.mean(mean_us)),
        "rmse": float(np.mean(rmses)),
        "iae": float(np.mean(iaes)),
        "recovery_time": float(np.mean(rec_times)),
        "changed": float(np.mean(changeds)),
        "bad": float(np.mean(bads)),
    }


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

    tol_over = 0.03
    tol_track = 0.05

    methods = ["PPO", "PPO+CONF"] + list(GENAI_CONFIGS.keys())
    rows = []

    print("\n=== Candidate Ablation Suite ===")
    print(
        f"{'Method':10s} | {'Sat%':>6s} | {'Mean rho':>9s} | "
        f"{'RMSE':>7s} | {'changed':>7s} | {'mean u':>7s}"
    )
    print("-" * 72)

    for method in methods:
        res = evaluate_method(
            method,
            model,
            tol_over=tol_over,
            tol_track=tol_track,
            n=N,
        )

        print(
            f"{method:10s} | {100*res['sat']:6.1f} | "
            f"{res['rho']:9.4f} | {res['rmse']:7.4f} | "
            f"{res['changed']:7.2f} | {res['mean_u']:7.4f}"
        )

        rows.append(
            {
                "method": method,
                "tol_over": tol_over,
                "tol_track": tol_track,
                "sat": res["sat"],
                "sat_percent": 100 * res["sat"],
                "rho": res["rho"],
                "mean_du": res["mean_du"],
                "max_du": res["max_du"],
                "mean_u": res["mean_u"],
                "rmse": res["rmse"],
                "iae": res["iae"],
                "recovery_time": res["recovery_time"],
                "changed": res["changed"],
                "bad": res["bad"],
            }
        )

    csv_path = os.path.join(OUT, "candidate_ablation_results.csv")
    json_path = os.path.join(OUT, "candidate_ablation_results.json")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "tol_over",
                "tol_track",
                "sat",
                "sat_percent",
                "rho",
                "mean_du",
                "max_du",
                "mean_u",
                "rmse",
                "iae",
                "recovery_time",
                "changed",
                "bad",
            ],
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