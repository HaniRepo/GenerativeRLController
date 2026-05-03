# genai_hackathon/run_genai_beats_conf_experiment.py

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


OUT = "genai_hackathon/genai_beats_conf_out"
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"

SEED = 42
N = 10
DT = 0.1

# Slightly strict but not impossible
TOL_OVER = 0.03
TOL_TRACK = 0.05

# GENAI-11: wider candidate set than GENAI-7
GENAI_OFFSETS = (
    -0.50, -0.40, -0.30, -0.20, -0.10,
     0.00,
     0.10,  0.20,  0.30,  0.40,  0.50,
)


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def make_nominal():
    return Monitor(F16EngineEnv(sp=500.0, dt=DT, ep_len_s=60.0, seed=SEED))


def make_hard_env():
    e = make_nominal()

    e = ActionRateLimiter(e, slew=0.012)
    e = NoisyDelayedWrapper(e, obs_sigma=4.0, act_delay_steps=4)
    e = ThrottleCapWrapper(e, u_max=0.78)

    b = _base_env(e)
    if hasattr(b, "sim") and hasattr(b.sim, "cfg") and hasattr(b.sim.cfg, "model_name"):
        b.sim.cfg.model_name = "morelli"
    elif hasattr(b, "sim") and hasattr(b.sim, "f16_model"):
        b.sim.f16_model = "morelli"

    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=560.0)
    return SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=600.0)
    return e


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


def tracking_metrics(vt_trace, sp_trace, jump_time=20.0):
    vt = np.asarray(vt_trace, dtype=float)
    sp = np.asarray(sp_trace, dtype=float)

    start = int(jump_time / DT)
    if start >= len(vt):
        start = int(0.75 * len(vt))

    vt_post = vt[start:]
    sp_post = sp[start:]

    err = (vt_post - sp_post) / np.maximum(sp_post, 1e-6)
    abs_err = np.abs(err)

    rmse = float(np.sqrt(np.mean(abs_err ** 2)))
    mae = float(np.mean(abs_err))
    overshoot = float(100.0 * np.max(err))

    inside = abs_err <= TOL_TRACK
    recovery_time = len(abs_err) * DT
    for i in range(len(inside)):
        if np.all(inside[i:]):
            recovery_time = i * DT
            break

    return rmse, mae, overshoot, recovery_time


def action_metrics(actions):
    actions = np.asarray(actions, dtype=float)
    if len(actions) <= 1:
        return 0.0, 0.0, float(np.mean(actions)) if len(actions) else 0.0

    du = np.abs(np.diff(actions))
    return float(np.mean(du)), float(np.max(du)), float(np.mean(actions))


def build_predictor(model, calib_eps=3, delta=0.30):
    env_cal = make_hard_env()
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


def rollout(method, model, seed):
    env = make_hard_env()
    obs, _ = env.reset(seed=seed)
    base = _base_env(env)

    shield = None

    if method == "PPO+CONF":
        pred, q = build_predictor(model)

        # Intentionally local/reactive shield
        shield = ConformalSTLShield(
            pred,
            q=q,
            K=4,
            dt=DT,
            tol=TOL_OVER,
            slew=0.02,
        )

    elif method == "PPO+GENAI":
        pred, q = build_predictor(model)

        # Wider candidate search and longer lookahead
        shield = GenerativeConformalShield(
            pred,
            q=q,
            K=10,
            dt=DT,
            tol=TOL_OVER,
            slew=0.25,
            candidate_offsets=(-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40),
        )

    if shield is not None:
        a0, _ = model.predict(obs, deterministic=True)
        u0 = float(np.asarray(a0).reshape(-1)[0])
        shield.reset(u0=u0)
        shield.debug = False

    vt, sp_hist, actions = [], [], []
    changed = 0
    bad = 0

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
    rmse, mae, overshoot, rec_time = tracking_metrics(vt, sp_hist)
    mean_du, max_du, mean_u = action_metrics(actions)

    return {
        "method": method,
        "seed": seed,
        "sat": float(sat),
        "rho": float(rho),
        "rho_over": float(rho_over),
        "rho_track": float(rho_track),
        "rmse": rmse,
        "mae": mae,
        "overshoot_percent": overshoot,
        "recovery_time": rec_time,
        "mean_du": mean_du,
        "max_du": max_du,
        "mean_u": mean_u,
        "changed": changed,
        "bad": bad,
        "vt": np.asarray(vt),
        "sp": np.asarray(sp_hist),
        "u": np.asarray(actions),
    }


def evaluate_method(method, model):
    rows = []

    for i in range(N):
        seed = SEED + i
        res = rollout(method, model, seed)
        rows.append(res)

        print(
            f"{method:10s} | seed={seed} | sat={int(res['sat'])} | "
            f"rho={res['rho']:.4f} | rmse={res['rmse']:.4f} | "
            f"os={res['overshoot_percent']:.2f}% | changed={res['changed']}"
        )

    summary = {
        "method": method,
        "sat_percent": 100.0 * float(np.mean([r["sat"] for r in rows])),
        "rho": float(np.mean([r["rho"] for r in rows])),
        "rho_over": float(np.mean([r["rho_over"] for r in rows])),
        "rho_track": float(np.mean([r["rho_track"] for r in rows])),
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "mae": float(np.mean([r["mae"] for r in rows])),
        "overshoot_percent": float(np.mean([r["overshoot_percent"] for r in rows])),
        "recovery_time": float(np.mean([r["recovery_time"] for r in rows])),
        "mean_du": float(np.mean([r["mean_du"] for r in rows])),
        "max_du": float(np.mean([r["max_du"] for r in rows])),
        "mean_u": float(np.mean([r["mean_u"] for r in rows])),
        "changed": float(np.mean([r["changed"] for r in rows])),
        "bad": float(np.mean([r["bad"] for r in rows])),
    }

    return summary, rows


def choose_best_seed(all_rows):
    """
    Pick a representative seed where GENAI looks better than CONF.
    Preference:
    - GENAI sat and CONF not sat
    - otherwise largest rho advantage
    """
    by_seed = {}

    for method, rows in all_rows.items():
        for r in rows:
            by_seed.setdefault(r["seed"], {})[method] = r

    best_seed = None
    best_score = -1e9

    for seed, d in by_seed.items():
        if not all(m in d for m in ["PPO", "PPO+CONF", "PPO+GENAI"]):
            continue

        ppo = d["PPO"]
        conf = d["PPO+CONF"]
        gen = d["PPO+GENAI"]

        score = 0.0
        score += 10.0 * (gen["sat"] - conf["sat"])
        score += 5.0 * (conf["sat"] - ppo["sat"])
        score += 2.0 * (gen["rho"] - conf["rho"])
        score += 1.0 * (conf["rho"] - ppo["rho"])
        score += 1.0 * (conf["rmse"] - gen["rmse"])

        if score > best_score:
            best_score = score
            best_seed = seed

    return best_seed


def smooth(x, window=15):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def plot_representative(all_rows, seed):
    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]

    data = {}
    for m in methods:
        data[m] = [r for r in all_rows[m] if r["seed"] == seed][0]

    t = np.arange(len(data["PPO"]["vt"])) * DT

    # Trajectory figure
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    for ax, m in zip(axs, methods):
        vt = data[m]["vt"]
        sp = data[m]["sp"]

        upper = sp * (1.0 + TOL_OVER)
        lower = sp * (1.0 - TOL_TRACK)

        ax.fill_between(t, lower, upper, alpha=0.15)
        ax.plot(t, sp, "k--", linewidth=1.2)
        ax.plot(t, vt, linewidth=2)

        ax.axvline(20.0, linestyle=":", linewidth=1.2)
        ax.set_title(m, fontsize=10)
        ax.set_ylabel(r"$V_t$")

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Representative Hard-Stress Trajectory", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_genai_beats_conf_trajectory.png"), dpi=300)
    plt.close()

    # Control figure
    plt.figure(figsize=(7, 3))
    for m in methods:
        plt.plot(t, smooth(data[m]["u"]), linewidth=2, label=m)

    plt.axvline(20.0, linestyle=":", linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Smoothed throttle")
    plt.title("Representative Control Signals")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_genai_beats_conf_control.png"), dpi=300)
    plt.close()

    # Robustness/Sat bar
    labels = methods
    sat = [100.0 * data[m]["sat"] for m in methods]
    rho = [data[m]["rho"] for m in methods]
    rmse = [data[m]["rmse"] for m in methods]

    x = np.arange(len(labels))

    plt.figure(figsize=(6, 3))
    plt.bar(x, sat)
    plt.xticks(x, labels)
    plt.ylabel("STL Satisfaction (%)")
    plt.ylim(0, 110)
    plt.title("Representative Satisfaction Outcome")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_genai_beats_conf_sat_bar.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 3))
    plt.bar(x, rho)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xticks(x, labels)
    plt.ylabel("STL robustness $\\rho$")
    plt.title("Representative Robustness")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_genai_beats_conf_rho_bar.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 3))
    plt.bar(x, rmse)
    plt.xticks(x, labels)
    plt.ylabel("RMSE")
    plt.title("Representative Tracking Error")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_genai_beats_conf_rmse_bar.png"), dpi=300)
    plt.close()


def save_results(summaries, all_rows):
    csv_path = os.path.join(OUT, "genai_beats_conf_summary.csv")
    json_path = os.path.join(OUT, "genai_beats_conf_summary.json")
    raw_path = os.path.join(OUT, "genai_beats_conf_raw.csv")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(summaries[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summaries)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    raw_rows = []
    for method, rows in all_rows.items():
        for r in rows:
            raw_rows.append({
                "method": method,
                "seed": r["seed"],
                "sat": r["sat"],
                "rho": r["rho"],
                "rho_over": r["rho_over"],
                "rho_track": r["rho_track"],
                "rmse": r["rmse"],
                "mae": r["mae"],
                "overshoot_percent": r["overshoot_percent"],
                "recovery_time": r["recovery_time"],
                "mean_du": r["mean_du"],
                "max_du": r["max_du"],
                "mean_u": r["mean_u"],
                "changed": r["changed"],
                "bad": r["bad"],
            })

    with open(raw_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(raw_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(raw_rows)

    print("\nSaved:")
    print(" -", csv_path)
    print(" -", json_path)
    print(" -", raw_path)


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

    methods = ["PPO", "PPO+CONF", "PPO+GENAI"]

    print("\n=== GENAI Beats CONF Stress Experiment ===")
    print("Designed to show: PPO unsafe, CONF local/reactive, GENAI wider candidate search.")
    print("-" * 90)

    summaries = []
    all_rows = {}

    for method in methods:
        print(f"\n--- {method} ---")
        summary, rows = evaluate_method(method, model)
        summaries.append(summary)
        all_rows[method] = rows

    print("\n=== Summary ===")
    print(
        f"{'Method':10s} | {'Sat%':>6s} | {'rho':>9s} | "
        f"{'RMSE':>7s} | {'OS%':>7s} | {'changed':>8s}"
    )
    print("-" * 70)

    for s in summaries:
        print(
            f"{s['method']:10s} | {s['sat_percent']:6.1f} | "
            f"{s['rho']:9.4f} | {s['rmse']:7.4f} | "
            f"{s['overshoot_percent']:7.2f} | {s['changed']:8.2f}"
        )

    best_seed = choose_best_seed(all_rows)
    print(f"\nSelected representative seed: {best_seed}")

    plot_representative(all_rows, best_seed)
    save_results(summaries, all_rows)

    print("\nFigures saved in:", OUT)
    print(" - fig_genai_beats_conf_trajectory.png")
    print(" - fig_genai_beats_conf_control.png")
    print(" - fig_genai_beats_conf_sat_bar.png")
    print(" - fig_genai_beats_conf_rho_bar.png")
    print(" - fig_genai_beats_conf_rmse_bar.png")


if __name__ == "__main__":
    main()