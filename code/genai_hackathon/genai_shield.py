# genai_hackathon/genai_shield.py

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from conformal_shield import worst_case_band_violation


class GenerativeConformalShield:
    """
    Lightweight GenAI shield:
    generate candidate actions around PPO output,
    roll them forward,
    select the candidate with best worst-case STL robustness.
    """

    def __init__(
        self,
        predictor,
        q,
        K=4,
        dt=0.1,
        tol=0.05,
        slew=0.03,
        candidate_offsets=(-0.10, -0.05, 0.0, 0.05, 0.10),
    ):
        self.p = predictor
        self.q = float(q)
        self.K = int(K)
        self.dt = float(dt)
        self.tol = float(tol)
        self.slew = float(slew)
        self.candidate_offsets = list(candidate_offsets)
        self.u_prev = 0.5
        #temp True
        self.debug = True

    def reset(self, u0=0.5):
        self.u_prev = float(u0)

    def _roll_pred(self, vt0, pw0, u_const):
        vt, pw = float(vt0), float(pw0)
        seq = []

        for _ in range(self.K):
            vt = self.p.predict_next(vt, pw, u_const)
            pw += 0.5 * (u_const - pw)
            seq.append(vt)

        return np.asarray(seq, dtype=float)

    def _build_candidates(self, u_rl, vt, sp):
        u_rl = float(np.clip(u_rl, 0.0, 1.0))

        candidates = []
        for off in self.candidate_offsets:
            u = np.clip(u_rl + off, 0.0, 1.0)

            # first action must satisfy slew feasibility
            u = np.clip(u, self.u_prev - self.slew, self.u_prev + self.slew)
            u = np.clip(u, 0.0, 1.0)

            candidates.append(float(u))

        # remove duplicates
        uniq = []
        seen = set()
        for u in candidates:
            key = round(u, 6)
            if key not in seen:
                seen.add(key)
                uniq.append(u)

        return uniq

    def filter(self, vt, pw, sp, u_rl):
        u_rl = float(u_rl)

        # evaluate PPO action first
        ppo_seq = self._roll_pred(vt, pw, u_rl)
        ppo_rho = worst_case_band_violation(ppo_seq, self.q, sp, tol=self.tol)

        candidates = self._build_candidates(u_rl, float(vt), float(sp))

        best_u = u_rl
        #best_rho = float(ppo_rho)
        best_rho = float(ppo_rho)
        best_score = -1e9

        for u in candidates:
            seq = self._roll_pred(vt, pw, u)
            rho = worst_case_band_violation(seq, self.q, sp, tol=self.tol)
            
            # safety-first score, with mild preference for tracking the setpoint
            final_err = abs((seq[-1] - sp) / max(1e-6, sp))
            score = float(rho - 0.02 * final_err)

            if "best_score" not in locals():
                best_score = -1e9

            if score > best_score:
                best_score = score
                best_rho = float(rho)
                best_u = float(u)
            '''
            if rho > best_rho:
                best_rho = float(rho)
                best_u = float(u)
          
        '''
            
        '''      
        # selective intervention: change only if clearly better
        if ppo_rho >=  0.0:
            chosen_u = u_rl
            chosen_rho = float(ppo_rho)
        elif best_rho > ppo_rho + 0.0005:
            chosen_u = best_u
            chosen_rho = best_rho
        else:
            chosen_u = u_rl
            chosen_rho = float(ppo_rho)
            '''
        # Active GenAI mode: choose the best candidate action
        chosen_u = best_u
        chosen_rho = best_rho

        if self.debug and ppo_rho < 0.0:
            print(
                f"[GENAI] u_rl={u_rl:.3f}, best_u={best_u:.3f}, "
                f"rho_rl={ppo_rho:.4f}, rho_best={best_rho:.4f}"
            )
        
        '''
        if self.debug and (abs(chosen_u - u_rl) >= 0.03 or chosen_rho < 0.0):
            print(
                f"[GENAI] vt={vt:.1f}, sp={sp:.1f}, "
                f"u_rl={u_rl:.3f}, u={chosen_u:.3f}, "
                f"rho_rl={ppo_rho:.4f}, rho_best={best_rho:.4f}"
            )
            '''

        self.u_prev = chosen_u
        return chosen_u, chosen_rho