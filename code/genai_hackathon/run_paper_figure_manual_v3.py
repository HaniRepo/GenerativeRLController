# genai_hackathon/run_paper_figure_manual.py

import os
import sys
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
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"
OUT = "genai_hackathon/paper_figures_out_v3"

SEED = 42
DT = 0.1
TOL_OVER = 0.03
TOL_TRACK = 0.05

GENAI_OFFSETS = (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30)  # GENAI-7


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
    if hasattr(b.sim, "cfg"):
        b.sim.cfg.model_name = "morelli"

    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=560.0)
    return e


# =========================
# PREDICTOR
# =========================
def build_predictor(model):
    env_cal = make_env()
    base = _base_env(env_cal)

    Vt, pw, thr = collect_calibration(base, model, episodes=3, seed=123)

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
def compute_metrics(vt, sp, actions):
    vt = np.asarray(vt)
    sp = np.asarray(sp)

    err = np.abs((vt - sp) / np.maximum(sp, 1e-6))

    rmse = float(np.sqrt(np.mean(err**2)))
    overshoot = float(np.max((vt - sp) / np.maximum(sp, 1e-6)))

    rho = float(np.min((sp * (1.03) - vt) / sp))
    sat = rho >= 0.0

    return rmse, overshoot, rho, sat


# =========================
# ROLLOUT
# =========================
def rollout(method, model):
    env = make_env()
    base = _base_env(env)

    pred, q = build_predictor(model)

    if method == "CONF":
        shield = ConformalSTLShield(pred, q=q, K=7, dt=DT, tol=TOL_OVER, slew=0.03)
    elif method == "GENAI":
        shield = GenerativeConformalShield(
            pred,
            q=q,
            K=10,
            dt=DT,
            tol=TOL_OVER,
            slew=0.25,
            candidate_offsets=GENAI_OFFSETS,
        )
    else:
        shield = None

    obs, _ = env.reset(seed=SEED)

    if shield:
        a0, _ = model.predict(obs, deterministic=True)
        u0 = float(np.asarray(a0).reshape(-1)[0])
        shield.reset(u0=u0)

    vt, sp, u = [], [], []
    changed = []

    done = False
    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_rl = float(np.asarray(a_rl).reshape(-1)[0])

        if shield:
            u_new, _ = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)
        else:
            u_new = u_rl

        changed.append(abs(u_new - u_rl) > 0.03)
        u.append(u_new)

        obs, _, done, trunc, info = env.step(np.array([u_new], dtype=np.float32))

        vt.append(info["Vt"])
        sp.append(info["sp"])

        if trunc:
            break

    rmse, overshoot, rho, sat = compute_metrics(vt, sp, u)

    print(f"{method:6s} | sat={sat} | rho={rho:.4f} | RMSE={rmse:.4f} | OS={overshoot:.4f}")

    return np.array(vt), np.array(sp), np.array(u), np.array(changed)


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

        upper = sp * (1 + TOL_OVER)
        lower = sp * (1 - TOL_TRACK)

        ax.fill_between(t, lower, upper, alpha=0.15)
        ax.plot(t, sp, "k--", linewidth=1.2)
        ax.plot(t, vt, linewidth=2)

        ax.axvline(20, linestyle=":", linewidth=1.2)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("$V_t$")

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "trajectory.png"), dpi=300)

    # -------- control --------
    plt.figure(figsize=(7, 3))
    plt.plot(t, ppo[2], label="PPO")
    plt.plot(t, conf[2], label="CONF")
    plt.plot(t, gen[2], label="GENAI")

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Throttle")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "control.png"), dpi=300)


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT, exist_ok=True)

    model = PPO.load(MODEL)

    print("\n=== Manual Figure Run ===")

    ppo = rollout("PPO", model)
    conf = rollout("CONF", model)
    gen = rollout("GENAI", model)

    plot_all(ppo, conf, gen)

    print("\nSaved in:", OUT)


if __name__ == "__main__":
    main()