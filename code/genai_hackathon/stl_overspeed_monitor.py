import numpy as np

def overspeed_spec(vt_trace, sp_trace, tol=0.03):
    """
    Safety STL:
    G( Vt <= sp * (1 + tol) )

    Returns satisfaction and robustness.
    Positive rho means safe.
    """
    vt_trace = np.asarray(vt_trace, dtype=float)
    sp_trace = np.asarray(sp_trace, dtype=float)

    margin = sp_trace * (1.0 + tol) - vt_trace
    rho = float(np.min(margin / np.maximum(sp_trace, 1e-6)))

    return rho >= 0.0, rho