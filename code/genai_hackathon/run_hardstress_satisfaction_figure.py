# genai_hackathon/run_hardstress_satisfaction_figure.py

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
from stress_wrappers import (
    SetpointJumpWrapper,
    NoisyDelayedWrapper,
    ThrottleCapWrapper,
    ActionRateLimiter,
)

from conformal_shield import (
    OneStepVTLinear,
    split_conformal_q,
    ConformalSTLShield,
    collect_calibration,
)
from genai_hackathon.genai_shield import GenerativeConformalShield


OUT = "genai_hackathon/hardstress_satisfaction_figure_out"
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"

SEED = 42
N = 10
DT = 0.1

TOL_OVER = 0.03
TOL_TRACK = 0.05

GENAI_OFFSETS = (-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40)


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def make_nominal(sp=500.0):
    return Monitor(F16EngineEnv(sp=sp, dt=DT, ep_len_s=60.0, seed=SEED))


def make_hardstress_env():
    e = make_nominal(sp=500.0)

    e = ActionRateLimiter(e, slew=0.012)
    e = NoisyDelayedWrapper(e, obs_sigma=4.0, act_delay_steps=4)
    e = ThrottleCapWrapper(e, u_max=0.78)

    b = _base_env(e)
    if hasattr(b, "sim") and hasattr(b.sim, "cfg") and hasattr(b.sim.cfg, "model_name"):
        b.sim.cfg.model_name = "morelli"
    elif hasattr(b, "sim") and hasattr(b.sim, "f16_model"):
        b.sim.f16_model = "morelli"

    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=560.0)
    return e


def spec(vt_trace, sp_trace, tol_over=TOL_OVER, tol_track=TOL_TRACK, last_frac=0.25):
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
    sat = int(rho >= 0.0)

    if sat:
        reason = "Satisfied"
    elif rho_over < rho_track:
        reason = "Overspeed"
    else:
        reason = "Tracking"

    return sat, rho, rho_over, rho_track, reason


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


def rollout(method, model, seed):
    env = make_hardstress_env()
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    shield = None

    if method == "PPO+CONF":
        pred, q = build_predictor(env, model)
        shield = ConformalSTLShield(pred, q=q, K=7, dt=DT, tol=TOL_OVER, slew=0.03)
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
            dt=DT,
            tol=TOL_OVER,
            slew=0.25,
            candidate_offsets=GENAI_OFFSETS,
        )
        a0, _ = model.predict(obs, deterministic=True)
        u0 = float(np.asarray(a0).reshape(-1)[0])
        shield.reset(u0=u0)
        shield.debug = False

    vt, sp_hist, actions = [], [], []
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

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))

        vt.append(info["Vt"])
        sp_hist.append(info["sp"])
        actions.append(u)

        if trunc:
            break

    sat, rho, rho_over, rho_track, reason = spec(vt, sp_hist)

    rmse = float(np.sqrt(np.mean(((np.asarray(vt) - np.asarray(sp_hist)) / np.maximum(sp_hist, 1e-6)) ** 2)))

    return {
        "method": method,
        "seed": seed,
        "sat": sat,
        "rho": rho,
        "rho_over": rho_over,
        "rho_track": rho_track,
        "reason": reason,
        "rmse": rmse,
        "changed": changed,
    }


def plot_robustness(rows):
    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]
    xpos = {m: i for i, m in enumerate(methods)}

    plt.figure(figsize=(7.2, 3.8))

    for r in rows:
        x = xpos[r["method"]]
        jitter = (r["seed"] - SEED - (N - 1) / 2.0) * 0.025

        marker = "o" if r["sat"] == 1 else "x"
        plt.scatter(x + jitter, r["rho"], s=60, marker=marker)

    plt.axhline(0.0, linestyle="--", linewidth=1.4, label=r"STL boundary $\rho=0$")

    plt.xticks(range(len(methods)), methods)
    plt.ylabel(r"STL robustness $\rho$")
    plt.xlabel("Method")
    plt.title("Hard-Stress Per-Seed STL Satisfaction")
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = os.path.join(OUT, "fig_hardstress_robustness_scatter.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_satisfaction_bar(rows):
    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]

    sat_percent = []
    mean_rho = []

    for m in methods:
        subset = [r for r in rows if r["method"] == m]
        sat_percent.append(100.0 * np.mean([r["sat"] for r in subset]))
        mean_rho.append(np.mean([r["rho"] for r in subset]))

    plt.figure(figsize=(6.8, 3.6))
    bars = plt.bar(methods, sat_percent)

    for b, val in zip(bars, sat_percent):
        plt.text(
            b.get_x() + b.get_width() / 2,
            val + 2,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylim(0, 115)
    plt.ylabel("STL satisfaction rate (%)")
    plt.xlabel("Method")
    plt.title("Hard-Stress Satisfaction Rate")
    plt.tight_layout()

    path = os.path.join(OUT, "fig_hardstress_satisfaction_bar.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_reason_stacked(rows):
    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]

    sat_vals, over_vals, track_vals = [], [], []

    for m in methods:
        subset = [r for r in rows if r["method"] == m]
        sat_vals.append(sum(r["reason"] == "Satisfied" for r in subset))
        over_vals.append(sum(r["reason"] == "Overspeed" for r in subset))
        track_vals.append(sum(r["reason"] == "Tracking" for r in subset))

    x = np.arange(len(methods))

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

    path = os.path.join(OUT, "fig_hardstress_violation_reason.png")
    plt.savefig(path, dpi=300)
    plt.close()


def save_csv(rows):
    csv_path = os.path.join(OUT, "hardstress_satisfaction_explain.csv")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "method",
            "seed",
            "sat",
            "rho",
            "rho_over",
            "rho_track",
            "reason",
            "rmse",
            "changed",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("Saved CSV:", csv_path)


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

    print("\n=== Hard-Stress Satisfaction Explanation ===")
    print(f"{'Method':10s} | {'Seed':>4s} | {'Sat':>3s} | {'rho':>8s} | {'Reason':>18s}")
    print("-" * 60)

    for method in methods:
        for i in range(N):
            seed = SEED + i
            r = rollout(method, model, seed)
            rows.append(r)

            print(
                f"{method:10s} | {seed:4d} | {r['sat']:3d} | "
                f"{r['rho']:8.4f} | {r['reason']:>18s}"
            )

    print("\nSummary:")
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        sat = 100.0 * np.mean([r["sat"] for r in subset])
        rho = np.mean([r["rho"] for r in subset])
        rmse = np.mean([r["rmse"] for r in subset])
        print(f"{method:10s}: Sat={sat:5.1f}% | mean rho={rho:.4f} | mean RMSE={rmse:.4f}")

    plot_robustness(rows)
    plot_satisfaction_bar(rows)
    plot_reason_stacked(rows)
    save_csv(rows)

    print("\nSaved figures:")
    print(" -", os.path.join(OUT, "fig_hardstress_robustness_scatter.png"))
    print(" -", os.path.join(OUT, "fig_hardstress_satisfaction_bar.png"))
    print(" -", os.path.join(OUT, "fig_hardstress_violation_reason.png"))


if __name__ == "__main__":
    main()