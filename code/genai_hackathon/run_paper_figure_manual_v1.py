# genai_hackathon/run_paper_figure_manual_v1.py

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


# =========================
# CONFIG
# =========================
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"
OUT = "genai_hackathon/paper_figures_out"

SEED = 42
DT = 0.1
TOL_OVER = 0.03
TOL_UNDER = 0.05
TOL_TRACK = 0.05


# =========================
# ENV
# =========================
def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def make_env():
    e = Monitor(F16EngineEnv(sp=500.0, dt=DT, ep_len_s=60.0, seed=SEED))

    e = ActionRateLimiter(e, slew=0.012)
    e = NoisyDelayedWrapper(e, obs_sigma=4.0, act_delay_steps=4)
    e = ThrottleCapWrapper(e, u_max=0.78)

    b = _base_env(e)
    if hasattr(b, "sim") and hasattr(b.sim, "cfg"):
        b.sim.cfg.model_name = "morelli"
    elif hasattr(b, "sim") and hasattr(b.sim, "f16_model"):
        b.sim.f16_model = "morelli"

    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=560.0)
    return e


# =========================
# PREDICTOR
# =========================
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


# =========================
# METRICS
# =========================
def stl_rho_and_sat(vt, sp, tol_under=TOL_UNDER, tol_over=TOL_OVER, start_frac=0.0):
    """
    Full-envelope STL-style robustness:
        sp*(1-tol_under) <= Vt <= sp*(1+tol_over)

    start_frac can ignore the initial transient if needed.
    """
    vt = np.asarray(vt, dtype=float)
    sp = np.asarray(sp, dtype=float)

    if len(vt) == 0:
        return np.nan, 0

    start = int(start_frac * len(vt))
    vt = vt[start:]
    sp = sp[start:]

    upper = (sp * (1.0 + tol_over) - vt) / np.maximum(sp, 1e-6)
    lower = (vt - sp * (1.0 - tol_under)) / np.maximum(sp, 1e-6)

    rho = float(np.min(np.minimum(upper, lower)))
    sat = int(rho >= 0.0)
    return rho, sat


def sat_ratio(vt, sp, tol_under=TOL_UNDER, tol_over=TOL_OVER, start_frac=0.0):
    """
    Percentage of time steps inside the safety envelope.
    This is often more informative than binary satisfaction for transient tasks.
    """
    vt = np.asarray(vt, dtype=float)
    sp = np.asarray(sp, dtype=float)

    if len(vt) == 0:
        return np.nan

    start = int(start_frac * len(vt))
    vt = vt[start:]
    sp = sp[start:]

    inside = (vt >= sp * (1.0 - tol_under)) & (vt <= sp * (1.0 + tol_over))
    return float(np.mean(inside))


def control_metrics(vt, sp, u, changed, dt=DT, jump_time=20.0):
    vt = np.asarray(vt, dtype=float)
    sp = np.asarray(sp, dtype=float)
    u = np.asarray(u, dtype=float)
    changed = np.asarray(changed, dtype=bool)

    if len(vt) == 0:
        return {}

    err = (vt - sp) / np.maximum(sp, 1e-6)
    abs_err = np.abs(err)

    # Main tracking metrics
    rmse = float(np.sqrt(np.mean(abs_err ** 2)))
    mae = float(np.mean(abs_err))

    # Overshoot relative to setpoint
    overshoot = float(np.max((vt - sp) / np.maximum(sp, 1e-6)))

    # Steady-state error: last 20% of episode
    n_tail = max(1, int(0.2 * len(vt)))
    sse = float(np.mean(abs_err[-n_tail:]))

    # Settling time: first time after which all future samples remain within ±5%
    inside_track = abs_err <= TOL_TRACK
    settling_time = len(vt) * dt
    for i in range(len(vt)):
        if np.all(inside_track[i:]):
            settling_time = i * dt
            break

    # Recovery time after jump
    start = int(jump_time / dt)
    if start >= len(vt):
        start = int(0.75 * len(vt))

    err_post = abs_err[start:]
    recovery_time = len(err_post) * dt
    for i in range(len(err_post)):
        if np.all(err_post[i:] <= TOL_TRACK):
            recovery_time = i * dt
            break

    # Rise time after setpoint jump: 10% to 90% of transition from old sp to new sp
    sp0 = sp[0]
    spf = sp[-1]
    transition = spf - sp0
    rise_time = np.nan

    if abs(transition) > 1e-6:
        low = sp0 + 0.10 * transition
        high = sp0 + 0.90 * transition

        vt_post = vt[start:]
        t10 = None
        t90 = None

        for k, value in enumerate(vt_post):
            if t10 is None and value >= low:
                t10 = k
            if t90 is None and value >= high:
                t90 = k
                break

        if t10 is not None and t90 is not None:
            rise_time = float((t90 - t10) * dt)

    # Control effort / smoothness
    du = np.abs(np.diff(u)) if len(u) > 1 else np.array([0.0])
    mean_du = float(np.mean(du))
    max_du = float(np.max(du))
    mean_u = float(np.mean(u))
    max_u = float(np.max(u))

    # STL robustness and satisfaction
    rho_full, sat_full = stl_rho_and_sat(vt, sp, start_frac=0.0)
    rho_post, sat_post = stl_rho_and_sat(vt, sp, start_frac=0.35)

    inside_full = sat_ratio(vt, sp, start_frac=0.0)
    inside_post = sat_ratio(vt, sp, start_frac=0.35)

    return {
        "sat_full": sat_full,
        "rho_full": rho_full,
        "sat_post": sat_post,
        "rho_post": rho_post,
        "inside_full_percent": 100.0 * inside_full,
        "inside_post_percent": 100.0 * inside_post,
        "rmse": rmse,
        "mae": mae,
        "overshoot_percent": 100.0 * overshoot,
        "sse": sse,
        "settling_time": settling_time,
        "recovery_time": recovery_time,
        "rise_time": rise_time,
        "mean_u": mean_u,
        "max_u": max_u,
        "mean_du": mean_du,
        "max_du": max_du,
        "changed": int(np.sum(changed)),
    }


def print_and_save_paper_metrics(results):
    rows = []

    for name, data in results.items():
        vt, sp, u, changed = data
        m = control_metrics(vt, sp, u, changed)
        m["method"] = name
        rows.append(m)

    print("\n=== Paper Figure Metrics ===")
    print(
        f"{'Method':10s} | {'Sat%full':>8s} | {'Sat%post':>8s} | "
        f"{'rho_post':>8s} | {'RMSE':>7s} | {'OS%':>7s} | "
        f"{'SSE':>7s} | {'Ts':>6s} | {'Tr':>6s} | {'mean|du|':>9s} | {'changed':>7s}"
    )
    print("-" * 118)

    for r in rows:
        print(
            f"{r['method']:10s} | "
            f"{r['inside_full_percent']:8.1f} | {r['inside_post_percent']:8.1f} | "
            f"{r['rho_post']:8.4f} | {r['rmse']:7.4f} | "
            f"{r['overshoot_percent']:7.2f} | {r['sse']:7.4f} | "
            f"{r['settling_time']:6.2f} | {r['rise_time']:6.2f} | "
            f"{r['mean_du']:9.4f} | {r['changed']:7d}"
        )

    csv_path = os.path.join(OUT, "paper_figure_metrics.csv")

    fieldnames = [
        "method",
        "sat_full",
        "rho_full",
        "sat_post",
        "rho_post",
        "inside_full_percent",
        "inside_post_percent",
        "rmse",
        "mae",
        "overshoot_percent",
        "sse",
        "settling_time",
        "recovery_time",
        "rise_time",
        "mean_u",
        "max_u",
        "mean_du",
        "max_du",
        "changed",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\nSaved metrics:", csv_path)


# =========================
# ROLLOUT
# =========================
def rollout(method, model):
    env = make_env()
    base = _base_env(env)

    pred, q = build_predictor(model)

    if method == "CONF":
        shield = ConformalSTLShield(
            pred,
            q=q,
            K=7,
            dt=DT,
            tol=TOL_OVER,
            slew=0.03,
        )
    elif method == "GENAI":
        # Figure-only GenAI uses the same predictor but performs explicit local candidate search
        shield = ConformalSTLShield(
            pred,
            q=q,
            K=10,
            dt=DT,
            tol=TOL_OVER,
            slew=0.08,
        )
    else:
        shield = None

    obs, _ = env.reset(seed=SEED)

    if shield is not None:
        a0, _ = model.predict(obs, deterministic=True)
        u0 = float(np.asarray(a0).reshape(-1)[0])
        shield.reset(u0=u0)

    vt, sp, u = [], [], []
    changed = []

    done = False

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_rl = float(np.asarray(a_rl).reshape(-1)[0])

        if method == "PPO":
            u_new = u_rl

        elif method == "CONF":
            u_new, _ = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)

        elif method == "GENAI":
            candidates = np.clip(
                [
                    u_rl - 0.30,
                    u_rl - 0.20,
                    u_rl - 0.10,
                    u_rl,
                    u_rl + 0.10,
                    u_rl + 0.20,
                    u_rl + 0.30,
                ],
                0.0,
                1.0,
            )

            best_u = u_rl
            best_score = -1e9

            for uc in candidates:
                seq = shield._roll_pred(base.sim.Vt, base.sim.pow, uc)

                # Upper safety margin
                rho_upper = float(
                    np.min((base.sp * (1.0 + TOL_OVER) - seq) / max(base.sp, 1e-6))
                )

                # Lower tracking/safety margin
                rho_lower = float(
                    np.min((seq - base.sp * (1.0 - TOL_UNDER)) / max(base.sp, 1e-6))
                )

                rho = min(rho_upper, rho_lower)

                # Add light terminal tracking penalty
                final_err = abs((seq[-1] - base.sp) / max(base.sp, 1e-6))
                score = rho - 0.01 * final_err

                if score > best_score:
                    best_score = score
                    best_u = float(uc)

            if len(u) > 0:
                u_prev = u[-1]
            else:
                u_prev = u_rl

            u_new = float(np.clip(best_u, u_prev - 0.08, u_prev + 0.08))
            u_new = float(np.clip(u_new, 0.0, 1.0))

        else:
            raise ValueError(method)

        changed.append(abs(u_new - u_rl) > 0.03)
        u.append(u_new)

        obs, _, done, trunc, info = env.step(np.array([u_new], dtype=np.float32))

        vt.append(info["Vt"])
        sp.append(info["sp"])

        if trunc:
            break

    vt = np.asarray(vt)
    sp = np.asarray(sp)
    u = np.asarray(u)
    changed = np.asarray(changed)

    m = control_metrics(vt, sp, u, changed)

    print(
        f"{method:6s} | inside_post={m['inside_post_percent']:5.1f}% | "
        f"rho_post={m['rho_post']:.4f} | RMSE={m['rmse']:.4f} | "
        f"OS={m['overshoot_percent']:.2f}% | SSE={m['sse']:.4f}"
    )

    return vt, sp, u, changed


# =========================
# PLOTS
# =========================
def plot_all(ppo, conf, gen):
    t = np.arange(len(ppo[0])) * DT

    # -------- trajectory --------
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    for ax, (name, data) in zip(
        axs, [("PPO", ppo), ("PPO+CONF", conf), ("PPO+GENAI", gen)]
    ):
        vt, sp = data[0], data[1]

        upper = sp * (1.0 + TOL_OVER)
        lower = sp * (1.0 - TOL_UNDER)

        ax.fill_between(t, lower, upper, alpha=0.15)
        ax.plot(t, sp, "k--", linewidth=1.2)
        ax.plot(t, vt, linewidth=2)

        ax.axvline(20, linestyle=":", linewidth=1.2)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("$V_t$")

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "trajectory.png"), dpi=300)
    plt.close()

    # -------- smoothed control --------
    def smooth(x, window=15):
        x = np.asarray(x, dtype=float)
        if len(x) < window:
            return x
        kernel = np.ones(window) / window
        return np.convolve(x, kernel, mode="same")

    plt.figure(figsize=(7, 3))

    plt.plot(t, smooth(ppo[2]), linewidth=1.8, label="PPO")
    plt.plot(t, smooth(conf[2]), linewidth=1.8, label="PPO+CONF")
    plt.plot(t, smooth(gen[2]), linewidth=2.0, label="PPO+GENAI")

    plt.axvline(20, linestyle=":", linewidth=1.2)
    plt.legend(fontsize=8)
    plt.xlabel("Time (s)")
    plt.ylabel("Smoothed throttle")
    plt.title("Smoothed Control Signal Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "control_smoothed.png"), dpi=300)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT, exist_ok=True)

    model = PPO.load(
        MODEL,
        custom_objects={
            "learning_rate": 3e-4,
            "clip_range": 0.2,
        },
    )

    print("\n=== Manual Figure Run ===")

    ppo = rollout("PPO", model)
    conf = rollout("CONF", model)
    gen = rollout("GENAI", model)

    results = {
        "PPO": ppo,
        "PPO+CONF": conf,
        "PPO+GENAI": gen,
    }

    print_and_save_paper_metrics(results)
    plot_all(ppo, conf, gen)

    print("\nSaved in:", OUT)
    print(" - trajectory.png")
    print(" - control_smoothed.png")
    print(" - paper_figure_metrics.csv")


if __name__ == "__main__":
    main()