#stl_monitor.py
import numpy as np

def settling_spec_last_window(vt_trace, sp, dt=0.1, window_s=10.0, tol=0.05):
    """
    STL-like check: require |Vt - sp|/sp <= tol over the last `window_s` seconds.
    Returns: (satisfied: bool, robustness: float), where robustness is min(tol - |err_rel|).
    """
    vt_trace = np.asarray(vt_trace, dtype=float)
    n = len(vt_trace)
    k_window = int(round(window_s / dt))
    if n == 0 or k_window <= 0:
        return False, -np.inf
    start = max(0, n - k_window)
    seg = vt_trace[start:]
    err_rel = np.abs((seg - sp) / max(1e-6, sp))
    margins = tol - err_rel
    rho = float(np.min(margins))
    return (rho >= 0.0), rho