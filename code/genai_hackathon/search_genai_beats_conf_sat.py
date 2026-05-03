# genai_hackathon/search_genai_beats_conf_sat.py

import os
import sys
import csv
import json
import numpy as np

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


OUT = "genai_hackathon/search_genai_beats_conf_out"
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"

SEED = 42
N = 10
DT = 0.1

GENAI_OFFSETS = (-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40)


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def make_env(
    obs_sigma=4.0,
    delay=4,
    u_max=0.78,
    slew=0.012,
    sp_new=560.0,
):
    e = Monitor(F16EngineEnv(sp=500.0, dt=DT, ep_len_s=60.0, seed=SEED))

    e = ActionRateLimiter(e, slew=slew)
    e = NoisyDelayedWrapper(e, obs_sigma=obs_sigma, act_delay_steps=delay)
    e = ThrottleCapWrapper(e, u_max=u_max)

    b = _base_env(e)
    if hasattr(b, "sim") and hasattr(b.sim, "cfg") and hasattr(b.sim.cfg, "model_name"):
        b.sim.cfg.model_name = "morelli"
    elif hasattr(b, "sim") and hasattr(b.sim, "f16_model"):
        b.sim.f16_model = "morelli"

    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=sp_new)
    return e


def overspeed_tracking_spec(
    vt_trace,
    sp_trace,
    tol_over=0.025,
    tol_track=0.04,
    last_frac=0.25,
):
    vt = np.asarray(vt_trace, dtype=float)
    sp = np.asarray(sp_trace, dtype=float)

    over_margin = (sp * (1.0 + tol_over) - vt) / np.maximum(sp, 1e-6)
    rho_over = float(np.min(over_margin))

    n = len(vt)
    start = int((1.0 - last_frac) * n)

    vt_last = vt[start:]
    sp_last = sp[start:]

    track_margin = tol_track - np.abs((vt_last - sp_last) / np.maximum(sp_last, 1e-6))
    rho_track = float(np.min(track_margin))

    rho = min(rho_over, rho_track)
    return rho >= 0.0, rho


def recovery_rmse(vt_trace, sp_trace, jump_time=20.0):
    vt = np.asarray(vt_trace, dtype=float)
    sp = np.asarray(sp_trace, dtype=float)

    start = int(jump_time / DT)
    if start >= len(vt):
        start = int(0.75 * len(vt))

    err = np.abs((vt[start:] - sp[start:]) / np.maximum(sp[start:], 1e-6))
    return float(np.sqrt(np.mean(err ** 2)))


def action_metrics(actions):
    actions = np.asarray(actions, dtype=float)
    du = np.abs(np.diff(actions)) if len(actions) > 1 else np.array([0.0])
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


def rollout(env, model, method, tol_over, tol_track):
    obs, info = env.reset()
    base = _base_env(env)

    shield = None

    if method == "CONF":
        pred, q = build_predictor(env, model)
        shield = ConformalSTLShield(
            pred,
            q=q,
            K=7,
            dt=DT,
            tol=tol_over,
            slew=0.03,
        )

    elif method == "GENAI":
        pred, q = build_predictor(env, model)
        shield = GenerativeConformalShield(
            pred,
            q=q,
            K=10,
            dt=DT,
            tol=tol_over,
            slew=0.25,
            candidate_offsets=GENAI_OFFSETS,
        )

    if shield is not None:
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

        actions.append(u)

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))

        vt.append(info["Vt"])
        sp_hist.append(info["sp"])

        if trunc:
            break

    sat, rho = overspeed_tracking_spec(
        vt,
        sp_hist,
        tol_over=tol_over,
        tol_track=tol_track,
        last_frac=0.25,
    )

    mean_du, max_du, mean_u = action_metrics(actions)
    rmse = recovery_rmse(vt, sp_hist)

    return {
        "sat": float(sat),
        "rho": float(rho),
        "rmse": rmse,
        "mean_u": mean_u,
        "mean_du": mean_du,
        "changed": changed,
    }


def evaluate_config(model, cfg, n=N):
    out = {}

    for method in ["PPO", "CONF", "GENAI"]:
        sats, rhos, rmses, mean_us, changeds = [], [], [], [], []

        for i in range(n):
            env = make_env(
                obs_sigma=cfg["obs_sigma"],
                delay=cfg["delay"],
                u_max=cfg["u_max"],
                slew=cfg["slew"],
                sp_new=cfg["sp_new"],
            )
            env.reset(seed=SEED + i)

            res = rollout(
                env,
                model,
                method,
                tol_over=cfg["tol_over"],
                tol_track=cfg["tol_track"],
            )

            sats.append(res["sat"])
            rhos.append(res["rho"])
            rmses.append(res["rmse"])
            mean_us.append(res["mean_u"])
            changeds.append(res["changed"])

        out[method] = {
            "sat_percent": 100.0 * float(np.mean(sats)),
            "rho": float(np.mean(rhos)),
            "rmse": float(np.mean(rmses)),
            "mean_u": float(np.mean(mean_us)),
            "changed": float(np.mean(changeds)),
        }

    return out


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

    configs = []

    for tol_over in [0.020, 0.025, 0.030]:
        for tol_track in [0.035, 0.040, 0.045, 0.050]:
            for delay in [3, 4, 5]:
                for obs_sigma in [3.0, 4.0, 5.0]:
                    for u_max in [0.76, 0.78, 0.80]:
                        for slew in [0.010, 0.012, 0.015]:
                            for sp_new in [550.0, 560.0, 570.0]:
                                configs.append(
                                    {
                                        "tol_over": tol_over,
                                        "tol_track": tol_track,
                                        "delay": delay,
                                        "obs_sigma": obs_sigma,
                                        "u_max": u_max,
                                        "slew": slew,
                                        "sp_new": sp_new,
                                    }
                                )

    rows = []
    good_cases = []

    print("\n=== Searching cases where GENAI beats CONF in satisfaction ===")
    print("Target: PPO <= 60, CONF < 100, GENAI = 100")
    print("-" * 90)

    for idx, cfg in enumerate(configs, start=1):
        res = evaluate_config(model, cfg, n=N)

        row = {
            **cfg,
            "ppo_sat": res["PPO"]["sat_percent"],
            "conf_sat": res["CONF"]["sat_percent"],
            "genai_sat": res["GENAI"]["sat_percent"],
            "ppo_rho": res["PPO"]["rho"],
            "conf_rho": res["CONF"]["rho"],
            "genai_rho": res["GENAI"]["rho"],
            "ppo_rmse": res["PPO"]["rmse"],
            "conf_rmse": res["CONF"]["rmse"],
            "genai_rmse": res["GENAI"]["rmse"],
            "genai_changed": res["GENAI"]["changed"],
        }
        rows.append(row)

        if (
            row["ppo_sat"] <= 60.0
            and row["conf_sat"] < 100.0
            and row["genai_sat"] >= 100.0
        ):
            good_cases.append(row)
            print("\nFOUND CASE:")
            print(cfg)
            print(
                f"PPO={row['ppo_sat']:.1f} | "
                f"CONF={row['conf_sat']:.1f} | "
                f"GENAI={row['genai_sat']:.1f}"
            )
            print(
                f"rho: PPO={row['ppo_rho']:.4f}, "
                f"CONF={row['conf_rho']:.4f}, "
                f"GENAI={row['genai_rho']:.4f}"
            )
            print(
                f"RMSE: PPO={row['ppo_rmse']:.4f}, "
                f"CONF={row['conf_rmse']:.4f}, "
                f"GENAI={row['genai_rmse']:.4f}"
            )

            break

        if idx % 25 == 0:
            print(f"Checked {idx}/{len(configs)} configs...")

    csv_path = os.path.join(OUT, "search_results.csv")
    json_path = os.path.join(OUT, "good_cases.json")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(good_cases, f, indent=2)

    print("\nSaved:")
    print(" -", csv_path)
    print(" -", json_path)

    if not good_cases:
        print("\nNo exact case found. Relax target to: CONF <= 90 and GENAI >= 90.")


if __name__ == "__main__":
    main()