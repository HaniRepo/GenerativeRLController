# genai_hackathon/run_incremental_severity_sweep.py

import os
import sys
import csv
import json
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


# =========================
# CONFIG
# =========================
OUT = "genai_hackathon/incremental_severity_sweep_out"
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"

SEED = 42
N = 10
DT = 0.1

TOL_OVER = 0.03
TOL_TRACK = 0.05

GENAI_OFFSETS = (-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40)


# =========================
# SEVERITY LEVELS
# =========================
SEVERITY_LEVELS = [
    {
        "level": 0,
        "name": "S0 Nominal",
        "sp_new": None,
        "obs_sigma": None,
        "delay": None,
        "u_max": None,
        "rate_slew": None,
        "mismatch": False,
    },
    {
        "level": 1,
        "name": "S1 Jump",
        "sp_new": 560.0,
        "obs_sigma": None,
        "delay": None,
        "u_max": None,
        "rate_slew": None,
        "mismatch": False,
    },
    {
        "level": 2,
        "name": "S2 Jump+Noise",
        "sp_new": 560.0,
        "obs_sigma": 2.0,
        "delay": 0,
        "u_max": None,
        "rate_slew": None,
        "mismatch": False,
    },
    {
        "level": 3,
        "name": "S3 +Delay",
        "sp_new": 560.0,
        "obs_sigma": 3.0,
        "delay": 2,
        "u_max": None,
        "rate_slew": None,
        "mismatch": False,
    },
    {
        "level": 4,
        "name": "S4 +ThrottleCap",
        "sp_new": 560.0,
        "obs_sigma": 3.0,
        "delay": 3,
        "u_max": 0.82,
        "rate_slew": None,
        "mismatch": False,
    },
    {
        "level": 5,
        "name": "S5 +RateLimit",
        "sp_new": 560.0,
        "obs_sigma": 4.0,
        "delay": 4,
        "u_max": 0.78,
        "rate_slew": 0.012,
        "mismatch": False,
    },
    {
        "level": 6,
        "name": "S6 +Mismatch",
        "sp_new": 560.0,
        "obs_sigma": 4.0,
        "delay": 4,
        "u_max": 0.78,
        "rate_slew": 0.012,
        "mismatch": True,
    },
]


# =========================
# ENV HELPERS
# =========================
def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def make_env(cfg):
    e = Monitor(F16EngineEnv(sp=500.0, dt=DT, ep_len_s=60.0, seed=SEED))

    if cfg["rate_slew"] is not None:
        e = ActionRateLimiter(e, slew=cfg["rate_slew"])

    if cfg["obs_sigma"] is not None:
        e = NoisyDelayedWrapper(
            e,
            obs_sigma=cfg["obs_sigma"],
            act_delay_steps=cfg["delay"] if cfg["delay"] is not None else 0,
        )

    if cfg["u_max"] is not None:
        e = ThrottleCapWrapper(e, u_max=cfg["u_max"])

    if cfg["mismatch"]:
        b = _base_env(e)
        if hasattr(b, "sim") and hasattr(b.sim, "cfg") and hasattr(b.sim.cfg, "model_name"):
            b.sim.cfg.model_name = "morelli"
        elif hasattr(b, "sim") and hasattr(b.sim, "f16_model"):
            b.sim.f16_model = "morelli"

    if cfg["sp_new"] is not None:
        e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=cfg["sp_new"])

    return e


# =========================
# METRICS
# =========================
def overspeed_tracking_spec(vt_trace, sp_trace, tol_over=TOL_OVER, tol_track=TOL_TRACK, last_frac=0.25):
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
    sat = rho >= 0.0

    return sat, rho, rho_over, rho_track


def tracking_rmse(vt_trace, sp_trace, jump_time=20.0):
    vt = np.asarray(vt_trace, dtype=float)
    sp = np.asarray(sp_trace, dtype=float)

    start = int(jump_time / DT)
    if start >= len(vt):
        start = int(0.5 * len(vt))

    err = (vt[start:] - sp[start:]) / np.maximum(sp[start:], 1e-6)
    return float(np.sqrt(np.mean(err ** 2)))


def action_metrics(actions):
    actions = np.asarray(actions, dtype=float)
    du = np.abs(np.diff(actions)) if len(actions) > 1 else np.array([0.0])

    return {
        "mean_u": float(np.mean(actions)) if len(actions) else 0.0,
        "mean_du": float(np.mean(du)),
        "max_du": float(np.max(du)),
    }


# =========================
# PREDICTOR
# =========================
def build_predictor(model, cfg, calib_eps=3, delta=0.30):
    env_cal = make_env(cfg)
    base = _base_env(env_cal)

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


# =========================
# ROLLOUT
# =========================
def rollout(method, model, cfg, seed):
    env = make_env(cfg)
    obs, _ = env.reset(seed=seed)
    base = _base_env(env)

    shield = None

    if method == "PPO+CONF":
        pred, q = build_predictor(model, cfg)
        shield = ConformalSTLShield(
            pred,
            q=q,
            K=7,
            dt=DT,
            tol=TOL_OVER,
            slew=0.03,
        )

    elif method == "PPO+GENAI":
        pred, q = build_predictor(model, cfg)
        shield = GenerativeConformalShield(
            pred,
            q=q,
            K=10,
            dt=DT,
            tol=TOL_OVER,
            slew=0.25,
            candidate_offsets=GENAI_OFFSETS,
        )

    if shield is not None:
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

        if method == "PPO":
            u = u_rl
            pred_rho = 0.0
        else:
            u, pred_rho = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)

        if abs(u - u_rl) >= 0.03:
            changed += 1
        if pred_rho < 0.0:
            bad += 1

        actions.append(u)

        obs, _, done, trunc, info = env.step(np.array([u], dtype=np.float32))

        vt.append(info["Vt"])
        sp_hist.append(info["sp"])

        if trunc:
            break

    sat, rho, rho_over, rho_track = overspeed_tracking_spec(vt, sp_hist)
    rmse = tracking_rmse(vt, sp_hist)
    am = action_metrics(actions)

    return {
        "sat": float(sat),
        "rho": float(rho),
        "rho_over": float(rho_over),
        "rho_track": float(rho_track),
        "rmse": rmse,
        "changed": changed,
        "bad": bad,
        **am,
    }


def evaluate_method(method, model, cfg):
    rows = []

    for i in range(N):
        seed = SEED + i
        res = rollout(method, model, cfg, seed)
        rows.append(res)

    return {
        "sat_percent": 100.0 * float(np.mean([r["sat"] for r in rows])),
        "rho": float(np.mean([r["rho"] for r in rows])),
        "rho_over": float(np.mean([r["rho_over"] for r in rows])),
        "rho_track": float(np.mean([r["rho_track"] for r in rows])),
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "mean_u": float(np.mean([r["mean_u"] for r in rows])),
        "mean_du": float(np.mean([r["mean_du"] for r in rows])),
        "max_du": float(np.mean([r["max_du"] for r in rows])),
        "changed": float(np.mean([r["changed"] for r in rows])),
        "bad": float(np.mean([r["bad"] for r in rows])),
    }


# =========================
# PLOTS
# =========================
def plot_metric(rows, metric, ylabel, filename, title):
    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]

    plt.figure(figsize=(7.2, 3.8))

    for method in methods:
        sub = [r for r in rows if r["method"] == method]
        sub = sorted(sub, key=lambda x: x["level"])

        x = [r["level"] for r in sub]
        y = [r[metric] for r in sub]

        plt.plot(x, y, marker="o", linewidth=2, label=method)

    if metric == "rho":
        plt.axhline(0.0, linestyle="--", linewidth=1.2)

    plt.xticks(
        [cfg["level"] for cfg in SEVERITY_LEVELS],
        [f"S{cfg['level']}" for cfg in SEVERITY_LEVELS],
    )
    plt.xlabel("Severity level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = os.path.join(OUT, filename)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_combined(rows):
    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    plot_specs = [
        ("sat_percent", "STL Sat. (%)", "Satisfaction"),
        ("rho", r"Mean $\rho$", "Robustness"),
        ("rmse", "RMSE", "Tracking Error"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, plot_specs):
        for method in methods:
            sub = [r for r in rows if r["method"] == method]
            sub = sorted(sub, key=lambda x: x["level"])

            x = [r["level"] for r in sub]
            y = [r[metric] for r in sub]

            ax.plot(x, y, marker="o", linewidth=2, label=method)

        if metric == "rho":
            ax.axhline(0.0, linestyle="--", linewidth=1.0)

        ax.set_xticks([cfg["level"] for cfg in SEVERITY_LEVELS])
        ax.set_xticklabels([f"S{cfg['level']}" for cfg in SEVERITY_LEVELS])
        ax.set_xlabel("Severity")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[0].legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(OUT, "fig_incremental_severity_combined.png")
    plt.savefig(path, dpi=300)
    plt.close()


def save_results(rows):
    csv_path = os.path.join(OUT, "incremental_severity_results.csv")
    json_path = os.path.join(OUT, "incremental_severity_results.json")

    fieldnames = [
        "level",
        "severity_name",
        "method",
        "sat_percent",
        "rho",
        "rho_over",
        "rho_track",
        "rmse",
        "mean_u",
        "mean_du",
        "max_du",
        "changed",
        "bad",
        "sp_new",
        "obs_sigma",
        "delay",
        "u_max",
        "rate_slew",
        "mismatch",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("\nSaved:")
    print(" -", csv_path)
    print(" -", json_path)


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT, exist_ok=True)

    env0 = Monitor(F16EngineEnv(sp=500.0, dt=DT, ep_len_s=60.0, seed=SEED))
    model = PPO.load(
        MODEL,
        env=env0,
        custom_objects={
            "learning_rate": 3e-4,
            "clip_range": 0.2,
        },
    )

    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]
    rows = []

    print("\n=== Incremental Severity Sweep ===")
    print(
        f"{'Level':>5s} | {'Severity':22s} | {'Method':10s} | "
        f"{'Sat%':>6s} | {'rho':>8s} | {'RMSE':>7s} | {'changed':>8s}"
    )
    print("-" * 90)

    for cfg in SEVERITY_LEVELS:
        for method in methods:
            res = evaluate_method(method, model, cfg)

            row = {
                "level": cfg["level"],
                "severity_name": cfg["name"],
                "method": method,
                **res,
                "sp_new": cfg["sp_new"],
                "obs_sigma": cfg["obs_sigma"],
                "delay": cfg["delay"],
                "u_max": cfg["u_max"],
                "rate_slew": cfg["rate_slew"],
                "mismatch": cfg["mismatch"],
            }
            rows.append(row)

            print(
                f"{cfg['level']:5d} | {cfg['name']:22s} | {method:10s} | "
                f"{res['sat_percent']:6.1f} | {res['rho']:8.4f} | "
                f"{res['rmse']:7.4f} | {res['changed']:8.2f}"
            )

    save_results(rows)

    plot_metric(
        rows,
        metric="sat_percent",
        ylabel="STL satisfaction (%)",
        filename="fig_severity_satisfaction.png",
        title="Satisfaction under Incremental Stress Severity",
    )

    plot_metric(
        rows,
        metric="rho",
        ylabel=r"Mean STL robustness $\rho$",
        filename="fig_severity_robustness.png",
        title="Robustness under Incremental Stress Severity",
    )

    plot_metric(
        rows,
        metric="rmse",
        ylabel="RMSE",
        filename="fig_severity_rmse.png",
        title="Tracking Error under Incremental Stress Severity",
    )

    plot_combined(rows)

    print("\nFigures saved in:", OUT)
    print(" - fig_severity_satisfaction.png")
    print(" - fig_severity_robustness.png")
    print(" - fig_severity_rmse.png")
    print(" - fig_incremental_severity_combined.png")


if __name__ == "__main__":
    main()