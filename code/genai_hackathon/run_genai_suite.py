# genai_hackathon/run_genai_suite.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# add parent /code folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os, csv, json
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window

from stress_wrappers import (
    ActionRateLimiter,
    NoisyDelayedWrapper,
    SetpointJumpWrapper,
    ThrottleCapWrapper
)

from conformal_shield import (
    OneStepVTLinear,
    split_conformal_q,
    collect_calibration
)

from genai_hackathon.genai_shield import GenerativeConformalShield


OUT = "genai_hackathon/out"
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


def make_stress():
    e = make_nominal()
    e = ActionRateLimiter(e, slew=0.015)
    e = NoisyDelayedWrapper(e, obs_sigma=6.0, act_delay_steps=2)
    e = ThrottleCapWrapper(e, u_max=0.81)
    e = SetpointJumpWrapper(e, t_jump_s=22.0, sp_new=595.0)
    return e


def eval_genai(env, model, tol, window_s, n=N):
    sats = []
    rhos = []

    base0 = _base_env(env)

    # calibration
    Vt, pw, thr = collect_calibration(
        base0,
        policy=model,
        episodes=6,
        random_throttle=False,
        seed=123
    )

    pred = OneStepVTLinear()
    pred.fit(Vt, pw, thr)

    pred_next = [
        pred.predict_next(v0, p0, u0)
        for v0, p0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])
    ]

    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=0.30)

    for _ in range(n):
        obs, info = env.reset(seed=SEED)

        base = _base_env(env)

        sh = GenerativeConformalShield(
            pred,
            q=q,
            K=6,
            dt=_dt(env),
            tol=tol,
            slew=0.03,
            candidate_offsets=(-0.14, -0.10, -0.06, -0.03, 0.0, 0.03, 0.06)
        )

        sh.reset(u0=0.5)

        vt = []

        done = False

        while not done:
            a_rl, _ = model.predict(obs, deterministic=True)

            u, rho = sh.filter(
                base.sim.Vt,
                base.sim.pow,
                base.sp,
                float(a_rl[0])
            )

            obs, r, done, trunc, info = env.step(
                np.array([u], dtype=np.float32)
            )

            vt.append(info["Vt"])

            if trunc:
                break

        sat, rho_final = settling_spec_last_window(
            np.asarray(vt),
            sp=base.sp,
            dt=_dt(env),
            window_s=window_s,
            tol=tol
        )

        sats.append(float(sat))
        rhos.append(float(rho_final))

    return float(np.mean(sats)), float(np.mean(rhos))


def main():
    os.makedirs(OUT, exist_ok=True)

    env = make_stress()

    model = PPO.load(
    MODEL,
    env=env,
    custom_objects={
        "learning_rate": 3e-4,
        "clip_range": 0.2
    }
)

    sat, rho = eval_genai(
        env,
        model,
        tol=0.02,
        window_s=5.0,
        n=N
    )

    print("\n=== GenAI Shield Results ===")
    print(f"STL Satisfaction: {100*sat:.1f}%")
    print(f"Mean Robustness: {rho:.4f}")


if __name__ == "__main__":
    main()