# genai_hackathon/run_satisfaction_explain_plot.py

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


OUT = "genai_hackathon/satisfaction_explain_out"
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"

SEED = 42
N = 10
DT = 0.1

TOL_OVER = 0.03
TOL_TRACK = 0.05


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def make_env():
    e = Monitor(F16EngineEnv(sp=500.0, dt=DT, ep_len_s=60.0, seed=SEED))
    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=560.0)
    return e


def overspeed_tracking_spec(vt_trace, sp_trace, tol_over=0.03, tol_track=0.05, last_frac=0.25):
    vt = np.asarray(vt_trace, dtype=float)
    sp = np.asarray(sp_trace, dtype=float)

    over_margin = (sp * (1.0 + tol_over) - vt) / np.maximum(sp, 1e-6)
    rho_over = float(np.min(over_margin))

    start = int((1.0 - last_frac) * len(vt))
    vt_last = vt[start:]
    sp_last = sp[start:]

    track_margin = tol_track - np.abs((vt_last - sp_last) / np.maximum(sp_last, 1e-6))
    rho_track = float(np.min(track_margin))

    rho = min(rho_over, rho_track)
    return rho >= 0.0, rho, rho_over, rho_track


def build_predictor(model):
    env_cal = make_env()
    base = _base_env(env_cal)

    Vt, pw, thr = collect_calibration(
        base,
        policy=model,
        episodes=3,
        random_throttle=False,
        seed=123,
    )

    pred = OneStepVTLinear()
    pred.fit(Vt, pw, thr)

    pred_next = [
        pred.predict_next(v0, p0, u0)
        for v0, p0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])
    ]

    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=0.30)
    return pred, q


def rollout(method, model, seed):
    env = make_env()
    obs, _ = env.reset(seed=seed)
    base = _base_env(env)

    shield = None

    if method == "PPO+CONF":
        pred, q = build_predictor(model)
        shield = ConformalSTLShield(pred, q=q, K=7, dt=DT, tol=TOL_OVER, slew=0.03)
        a0, _ = model.predict(obs, deterministic=True)
        shield.reset(u0=float(np.asarray(a0).reshape(-1)[0]))
        shield.debug = False

    elif method == "PPO+GENAI":
        pred, q = build_predictor(model)
        shield = GenerativeConformalShield(
            pred,
            q=q,
            K=10,
            dt=DT,
            tol=TOL_OVER,
            slew=0.25,
            candidate_offsets=(-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30),
        )
        a0, _ = model.predict(obs, deterministic=True)
        shield.reset(u0=float(np.asarray(a0).reshape(-1)[0]))
        shield.debug = False

    vt, sp, actions = [], [], []
    changed = 0
    done = False

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_rl = float(np.asarray(a_rl).reshape(-1)[0])

        if method == "PPO":
            u = u_rl
        else:
            u, _ = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)

        if abs(u - u_rl) >= 0.03:
            changed += 1

        obs, _, done, trunc, info = env.step(np.array([u], dtype=np.float32))

        vt.append(info["Vt"])
        sp.append(info["sp"])
        actions.append(u)

        if trunc:
            break

    sat, rho, rho_over, rho_track = overspeed_tracking_spec(vt, sp)

    return {
        "method": method,
        "seed": seed,
        "sat": int(sat),
        "rho": rho,
        "rho_over": rho_over,
        "rho_track": rho_track,
        "changed": changed,
        "vt": np.asarray(vt),
        "sp": np.asarray(sp),
        "u": np.asarray(actions),
    }


def explain_failure(row):
    if row["sat"] == 1:
        return "Satisfied"

    if row["rho_over"] < row["rho_track"]:
        return "Overspeed violation"
    return "Tracking violation"


def plot_robustness(rows):
    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]
    x_positions = {m: i for i, m in enumerate(methods)}

    plt.figure(figsize=(7.0, 3.8))

    for row in rows:
        x = x_positions[row["method"]]
        jitter = (row["seed"] - SEED - (N - 1) / 2) * 0.025

        marker = "o" if row["sat"] == 1 else "x"
        plt.scatter(
            x + jitter,
            row["rho"],
            s=55,
            marker=marker,
        )

    plt.axhline(0.0, linestyle="--", linewidth=1.4, label="STL satisfaction boundary ($\\rho=0$)")

    plt.xticks(range(len(methods)), methods)
    plt.ylabel("STL robustness $\\rho$")
    plt.xlabel("Method")
    plt.title("Per-Seed STL Satisfaction Explanation")
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(os.path.join(OUT, "fig_satisfaction_robustness_scatter.png"), dpi=300)
    plt.close()


def plot_violation_reason(rows):
    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]

    data = {}
    for m in methods:
        subset = [r for r in rows if r["method"] == m]
        sat = sum(r["sat"] for r in subset)
        overspeed = sum(1 for r in subset if r["sat"] == 0 and explain_failure(r) == "Overspeed violation")
        tracking = sum(1 for r in subset if r["sat"] == 0 and explain_failure(r) == "Tracking violation")
        data[m] = [sat, overspeed, tracking]

    x = np.arange(len(methods))
    sat_vals = [data[m][0] for m in methods]
    over_vals = [data[m][1] for m in methods]
    track_vals = [data[m][2] for m in methods]

    plt.figure(figsize=(7.0, 3.8))
    plt.bar(x, sat_vals, label="Satisfied")
    plt.bar(x, over_vals, bottom=sat_vals, label="Overspeed violation")
    bottom2 = np.asarray(sat_vals) + np.asarray(over_vals)
    plt.bar(x, track_vals, bottom=bottom2, label="Tracking violation")

    plt.xticks(x, methods)
    plt.ylabel("Number of seeds")
    plt.xlabel("Method")
    plt.title("Why Episodes Satisfy or Violate the STL Specification")
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(os.path.join(OUT, "fig_satisfaction_reason_stackedbar.png"), dpi=300)
    plt.close()


def save_rows(rows):
    csv_path = os.path.join(OUT, "satisfaction_explain_results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "method",
            "seed",
            "sat",
            "rho",
            "rho_over",
            "rho_track",
            "changed",
            "reason",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for r in rows:
            w.writerow({
                "method": r["method"],
                "seed": r["seed"],
                "sat": r["sat"],
                "rho": r["rho"],
                "rho_over": r["rho_over"],
                "rho_track": r["rho_track"],
                "changed": r["changed"],
                "reason": explain_failure(r),
            })

    print("Saved:", csv_path)


def main():
    os.makedirs(OUT, exist_ok=True)

    model = PPO.load(
        MODEL,
        custom_objects={
            "learning_rate": 3e-4,
            "clip_range": 0.2,
        },
    )

    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]
    rows = []

    print("\n=== Satisfaction Explanation Suite ===")
    print(f"{'Method':10s} | {'Seed':>4s} | {'Sat':>3s} | {'rho':>8s} | {'rho_over':>8s} | {'rho_track':>9s} | Reason")
    print("-" * 86)

    for method in methods:
        for i in range(N):
            seed = SEED + i
            r = rollout(method, model, seed)
            rows.append(r)

            print(
                f"{method:10s} | {seed:4d} | {r['sat']:3d} | "
                f"{r['rho']:8.4f} | {r['rho_over']:8.4f} | "
                f"{r['rho_track']:9.4f} | {explain_failure(r)}"
            )

    plot_robustness(rows)
    plot_violation_reason(rows)
    save_rows(rows)

    print("\nFigures saved:")
    print(" -", os.path.join(OUT, "fig_satisfaction_robustness_scatter.png"))
    print(" -", os.path.join(OUT, "fig_satisfaction_reason_stackedbar.png"))


if __name__ == "__main__":
    main()