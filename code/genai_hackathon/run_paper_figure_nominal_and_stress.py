# genai_hackathon/run_paper_figure_nominal_and_stress.py

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


MODEL = "shield_pack/ppo_f16_engine_baseline.zip"
OUT = "genai_hackathon/paper_figures_nominal_stress"

SEED = 42
DT = 0.1
TOL_OVER = 0.03
TOL_UNDER = 0.05
TOL_TRACK = 0.05


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def make_nominal_env():
    # True nominal: no jump, no noise, no delay, no mismatch, no actuator stress
    return Monitor(F16EngineEnv(sp=500.0, dt=DT, ep_len_s=60.0, seed=SEED))


def make_stress_env():
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


def build_predictor(model, env_fn):
    env_cal = env_fn()
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


def compute_metrics(vt, sp, u, changed):
    vt = np.asarray(vt, dtype=float)
    sp = np.asarray(sp, dtype=float)
    u = np.asarray(u, dtype=float)
    changed = np.asarray(changed, dtype=bool)

    err = (vt - sp) / np.maximum(sp, 1e-6)
    abs_err = np.abs(err)

    rmse = float(np.sqrt(np.mean(abs_err ** 2)))
    mae = float(np.mean(abs_err))
    overshoot = float(100.0 * np.max((vt - sp) / np.maximum(sp, 1e-6)))

    upper = (sp * (1.0 + TOL_OVER) - vt) / np.maximum(sp, 1e-6)
    lower = (vt - sp * (1.0 - TOL_UNDER)) / np.maximum(sp, 1e-6)
    rho = float(np.min(np.minimum(upper, lower)))
    sat = int(rho >= 0.0)

    inside = (vt >= sp * (1.0 - TOL_UNDER)) & (vt <= sp * (1.0 + TOL_OVER))
    inside_percent = float(100.0 * np.mean(inside))

    du = np.abs(np.diff(u)) if len(u) > 1 else np.array([0.0])

    return {
        "sat": sat,
        "rho": rho,
        "inside_percent": inside_percent,
        "rmse": rmse,
        "mae": mae,
        "overshoot_percent": overshoot,
        "mean_u": float(np.mean(u)),
        "mean_du": float(np.mean(du)),
        "max_du": float(np.max(du)),
        "changed": int(np.sum(changed)),
    }


def rollout(method, model, env_fn, scenario_name):
    env = env_fn()
    base = _base_env(env)

    pred, q = build_predictor(model, env_fn)

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
        # Figure GenAI: candidate-search controller using same conformal predictor
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

                rho_upper = float(
                    np.min((base.sp * (1.0 + TOL_OVER) - seq) / max(base.sp, 1e-6))
                )
                rho_lower = float(
                    np.min((seq - base.sp * (1.0 - TOL_UNDER)) / max(base.sp, 1e-6))
                )
                rho = min(rho_upper, rho_lower)

                terminal_err = abs((seq[-1] - base.sp) / max(base.sp, 1e-6))
                score = rho - 0.01 * terminal_err

                if score > best_score:
                    best_score = score
                    best_u = float(uc)

            u_prev = u[-1] if len(u) > 0 else u_rl
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

    m = compute_metrics(vt, sp, u, changed)

    print(
        f"{scenario_name:8s} | {method:6s} | "
        f"sat={m['sat']} | inside={m['inside_percent']:5.1f}% | "
        f"rho={m['rho']:.4f} | RMSE={m['rmse']:.4f} | "
        f"OS={m['overshoot_percent']:.2f}% | changed={m['changed']}"
    )

    return vt, sp, u, changed, m


def smooth(x, window=15):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def plot_scenario(results, scenario_name, filename_prefix, show_jump=False):
    ppo = results["PPO"]
    conf = results["CONF"]
    gen = results["GENAI"]

    t = np.arange(len(ppo[0])) * DT

    # Trajectories
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    for ax, label, data in zip(
        axs,
        ["PPO", "PPO+CONF", "PPO+GENAI"],
        [ppo, conf, gen],
    ):
        vt, sp = data[0], data[1]
        upper = sp * (1.0 + TOL_OVER)
        lower = sp * (1.0 - TOL_UNDER)

        ax.fill_between(t, lower, upper, alpha=0.15, label="Safety envelope")
        ax.plot(t, sp, "k--", linewidth=1.2, label="Setpoint")
        ax.plot(t, vt, linewidth=2, label=label)

        if show_jump:
            ax.axvline(20, linestyle=":", linewidth=1.2)

        ax.set_title(label, fontsize=10)
        ax.set_ylabel(r"$V_t$")

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{scenario_name}: Airspeed Trajectories", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"{filename_prefix}_trajectory.png"), dpi=300)
    plt.close()

    # Smoothed controls
    plt.figure(figsize=(7, 3))
    plt.plot(t, smooth(ppo[2]), linewidth=1.8, label="PPO")
    plt.plot(t, smooth(conf[2]), linewidth=1.8, label="PPO+CONF")
    plt.plot(t, smooth(gen[2]), linewidth=2.0, label="PPO+GENAI")

    if show_jump:
        plt.axvline(20, linestyle=":", linewidth=1.2)

    plt.xlabel("Time (s)")
    plt.ylabel("Smoothed throttle")
    plt.title(f"{scenario_name}: Smoothed Control Signals")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"{filename_prefix}_control_smoothed.png"), dpi=300)
    plt.close()


def save_metrics(all_results):
    rows = []

    for scenario_name, scenario_results in all_results.items():
        for method_name, data in scenario_results.items():
            row = {"scenario": scenario_name, "method": method_name}
            row.update(data[4])
            rows.append(row)

    csv_path = os.path.join(OUT, "nominal_stress_metrics.csv")

    fieldnames = [
        "scenario",
        "method",
        "sat",
        "rho",
        "inside_percent",
        "rmse",
        "mae",
        "overshoot_percent",
        "mean_u",
        "mean_du",
        "max_du",
        "changed",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\nSaved metrics:", csv_path)


def main():
    os.makedirs(OUT, exist_ok=True)

    model = PPO.load(
        MODEL,
        custom_objects={
            "learning_rate": 3e-4,
            "clip_range": 0.2,
        },
    )

    scenarios = {
        "Nominal": (make_nominal_env, False, "nominal"),
        "Stress": (make_stress_env, True, "stress"),
    }

    all_results = {}

    print("\n=== Nominal and Stress Figure Run ===")
    print("Scenario | Method | Metrics")
    print("-" * 90)

    for scenario_name, (env_fn, show_jump, prefix) in scenarios.items():
        scenario_results = {}

        for method in ["PPO", "CONF", "GENAI"]:
            scenario_results[method] = rollout(
                method=method,
                model=model,
                env_fn=env_fn,
                scenario_name=scenario_name,
            )

        all_results[scenario_name] = scenario_results
        plot_scenario(
            scenario_results,
            scenario_name=scenario_name,
            filename_prefix=prefix,
            show_jump=show_jump,
        )

    save_metrics(all_results)

    print("\nSaved figures in:", OUT)
    print(" - nominal_trajectory.png")
    print(" - nominal_control_smoothed.png")
    print(" - stress_trajectory.png")
    print(" - stress_control_smoothed.png")


if __name__ == "__main__":
    main()