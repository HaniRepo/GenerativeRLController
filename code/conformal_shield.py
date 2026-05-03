# conformal_shield.py
import numpy as np

# --- tiny one-step predictor (linear) ---
class OneStepVTLinear:
    """Predict Vt_{t+1} from [Vt_t, pow_t, thr_t, 1]."""
    def __init__(self): self.coef_ = None

    def fit(self, Vt, pow_, thr):
        Vt, pow_, thr = map(lambda a: np.asarray(a, float), (Vt, pow_, thr))
        X = np.stack([Vt[:-1], pow_[:-1], thr[:-1], np.ones_like(thr[:-1])], axis=1)
        y = Vt[1:]
        self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)

    def predict_next(self, vt, pw, thr):
        a,b,c,d = self.coef_
        return float(a*vt + b*pw + c*thr + d)

def split_conformal_q(residuals, delta=0.10):
    r = np.abs(np.asarray(residuals, float))
    return float(np.quantile(r, 1.0 - delta, method="higher"))

def worst_case_band_violation(pred_seq, q, sp, tol=0.05):
    pred_seq = np.asarray(pred_seq, float)
    err_up = np.abs((pred_seq + q - sp) / max(1e-6, sp))
    err_dn = np.abs((pred_seq - q - sp) / max(1e-6, sp))
    #margins = tol - np.minimum(err_up, err_dn) Fixed issue 1
    #margins = tol - np.minimum(err_up, err_dn)
    margins = tol - np.maximum(err_up, err_dn)
    

    
    return float(np.min(margins))  # < 0 => predicted violation

# --- the shield ---
class ConformalSTLShield:
    """
    Predict K steps ahead with a one-step model + conformal band (±q).
    If worst-case robustness over horizon is negative, adjust throttle toward setpoint.
    """
    def __init__(self, predictor, q, K=20, dt=0.1, tol=0.05, slew=0.05):
        self.p, self.q = predictor, float(q)
        self.K, self.dt, self.tol = int(K), float(dt), float(tol)
        self.slew = float(slew)
        self.u_prev = 0.5

        # ADDED for LOG
        self.debug = False
        self.debug_eps = 0.05



    def reset(self, u0=0.5): self.u_prev = float(u0)

    def _roll_pred(self, vt0, pw0, u_const):
        vt, pw = float(vt0), float(pw0)
        seq = []
        for _ in range(self.K):
            vt = self.p.predict_next(vt, pw, u_const)
            # crude pow update: towards u (engine lag captured implicitly in one-step fit)
            pw += 0.5 * (u_const - pw)   # small lag factor; improves multi-step stability
            
            seq.append(vt)
        return np.array(seq)

    def filter(self, vt, pw, sp, u_rl):
        #return float(u_rl), 999.0
        # predictive check around constant action u_rl
        seq = self._roll_pred(vt, pw, u_rl)
        rho_worst = worst_case_band_violation(seq, self.q, sp, tol=self.tol)
        '''
        if rho_worst >= 0.0:  # it was -0.2
            u_tgt = u_rl
        else:
            # nudge toward setpoint if violation predicted
            bias = 0.15 if vt < sp else -0.15
            u_tgt = np.clip(u_rl + bias, 0.0, 1.0)
        '''
        cands = np.clip([u_rl-0.15, u_rl-0.08, u_rl, u_rl+0.08, u_rl+0.15], 0.0, 1.0)
        best_u = u_rl
        best_rho = -1e9

        for u_c in cands:
            seq = self._roll_pred(vt, pw, u_c)
            rho = worst_case_band_violation(seq, self.q, sp, tol=self.tol)
            if rho > best_rho:
                best_rho = rho
                best_u = u_c
                
        if rho_worst >= -0.005:
            u_tgt = u_rl
        elif best_rho > rho_worst + 0.002:
            u_tgt = best_u
        else:
            u_tgt = u_rl
        '''    
        #u_tgt = best_u
        if best_rho >= 0.005:
            u_tgt = u_rl   # stay with RL if safe
        else:
            u_tgt = best_u 
        '''
        # slew-limit
        u = np.clip(u_tgt, self.u_prev - self.slew, self.u_prev + self.slew)
        u = float(np.clip(u, 0.0, 1.0))
        self.u_prev = u
        # debug log (optional)
        if self.debug and (best_rho < 0.01 or abs(u - u_rl) >= 0.05):
            print(f"[CONF] vt={vt:.1f}, sp={sp:.1f}, u_rl={u_rl:.3f}, u={u:.3f}, rho={best_rho:.4f}")

        return u, best_rho

# --- calibration utility ---
def collect_calibration(env, policy=None, episodes=10, random_throttle=False, seed=123):
    """
    Roll out to gather (Vt, pow, thr) and residuals for conformal q.
    If policy is None or random_throttle=True, use noisy random actions in [0,1].
    """
    rng = np.random.default_rng(seed)
    Vt_all, pow_all, thr_all = [], [], []
    from stable_baselines3.common.base_class import BaseAlgorithm

    for _ in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0,1e9)))
        done = False
        while not done:
            if random_throttle or policy is None:
                u = float(np.clip(rng.uniform(0.0, 1.0) + 0.1*rng.normal(), 0.0, 1.0))
                action = np.array([u], dtype=np.float32)
            else:
                action, _ = policy.predict(obs, deterministic=False)
                u = float(np.clip(action[0], 0.0, 1.0))

            # record before step
            vt0 = env.sim.Vt; pw0 = env.sim.pow; thr0 = u
            obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
            vt1 = env.sim.Vt

            Vt_all.append(vt0); pow_all.append(pw0); thr_all.append(thr0)
            # store residual as |pred(vt0,pw0,thr0) - vt1| later after fit
            if trunc: break

    Vt_all, pow_all, thr_all = map(np.asarray, (Vt_all, pow_all, thr_all))
    return Vt_all, pow_all, thr_all
