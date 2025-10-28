from numba import njit
from numpy.linalg import norm
from numpy import array, concatenate, inf, float64, expand_dims, log10
from numpy.typing import NDArray
import numpy as np
from typing import List, Tuple, Callable
import dynamicslib.DOP853_coefs as coefs


# largely written by chatgpt. I modified.
@njit(cache=True)
def rkf45(
    func: Callable,
    x0: NDArray,
    t_span: Tuple[float, float],
    atol: float = 1e-10,
    rtol: float = 1e-10,
    init_step: float = 1e-6,
):

    t0, tf = t_span
    t = t0
    xs = expand_dims(x0, axis=0)
    ts = array([t0], dtype=float64)
    x = x0
    h = init_step

    while t < tf:
        if t + h > tf:
            h = tf - t
        tol = atol + norm(x, inf) * rtol
        # fmt: off
        k1 = h * func(t, x)
        k2 = h * func(t + (1/4)*h, x + (1/4)*k1)
        k3 = h * func(t + (3/8)*h, x + (3/32)*k1 + (9/32)*k2)
        k4 = h * func(t + (12/13)*h, x + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
        k5 = h * func(t + h, x + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
        k6 = h * func(t + (1/2)*h, x - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)
        x4 = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4104) * k4 - (1 / 5) * k5
        x5 = x + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9 / 50)*k5 + (2 / 55)*k6
        # fmt: on

        error = norm(x5 - x4)

        # no div0
        s = (tol / error) ** 0.25 if error != 0 else 2

        if s > 2:
            s = 2
        if s < 0.1:
            s = 0.1

        if error <= tol:
            t += h
            x = x5
            ts = concatenate((ts, array([t], dtype=float64)))
            xs = concatenate((xs, expand_dims(x, axis=0)))

        h *= s

    return ts, xs


# Coefficients from the book cited by scipy docs. Do not use.
# @njit(cache=True)
# def rkf78(
#     func: Callable,
#     x0: NDArray,
#     t_span: Tuple[float, float],
#     atol: float = 1e-10,
#     rtol: float = 1e-10,
#     init_step: float = 1e-6,
# ):
#     atol *= 1e5
#     rtol *= 1e5
#     t0, tf = t_span
#     t = t0
#     xs = expand_dims(x0, axis=0)
#     ts = array([t0], dtype=float64)
#     x = x0
#     h = init_step

#     # pp180 of RKEM
#     while t < tf:
#         if t + h > tf:
#             h = tf - t
#         tol = atol + norm(x, inf) * rtol
#         # fmt: off
#         k1  = h * func(t           , x)
#         k2  = h * func(t + (2/27)*h, x + (    2/27  )*k1)
#         k3  = h * func(t + (1/9 )*h, x + (    1/36  )*k1 + (1/12)*k2)
#         k4  = h * func(t + (1/6 )*h, x + (    1/24  )*k1 + ( 0  )*k2 + (  1/8 )*k3)
#         k5  = h * func(t + (5/12)*h, x + (    5/12  )*k1 + ( 0  )*k2 + (-25/16)*k3 + (  25/16 )*k4)
#         k6  = h * func(t + (1/2 )*h, x + (    1/20  )*k1 + ( 0  )*k2 + (   0  )*k3 + (   1/4  )*k4 + (   1/5   )*k5)
#         k7  = h * func(t + (5/6 )*h, x + (  -25/108 )*k1 + ( 0  )*k2 + (   0  )*k3 + ( 125/108)*k4 + ( -65/27  )*k5 + ( 125/54 )*k6)
#         k8  = h * func(t + (1/6 )*h, x + (   31/300 )*k1 + ( 0  )*k2 + (   0  )*k3 + (    0   )*k4 + (  61/225 )*k5 + (  -2/9  )*k6 + (  13/900 )*k7)
#         k9  = h * func(t + (2/3 )*h, x + (     2    )*k1 + ( 0  )*k2 + (   0  )*k3 + ( -53/6  )*k4 + ( 704/45  )*k5 + (-107/9  )*k6 + (  67/90  )*k7 + (  3   )*k8)
#         k10 = h * func(t + (1/3 )*h, x + (  -91/108 )*k1 + ( 0  )*k2 + (   0  )*k3 + (  23/108)*k4 + (-976/135 )*k5 + ( 311/54 )*k6 + ( -19/60  )*k7 + (17/6  )*k8 + (-1/12 )*k9)
#         k11 = h * func(t + ( 1  )*h, x + ( 2383/4100)*k1 + ( 0  )*k2 + (   0  )*k3 + (-341/164)*k4 + (4496/1025)*k5 + (-301/82 )*k6 + (2133/4100)*k7 + (45/82 )*k8 + (45/164)*k9 + (18/41 )*k10)
#         k12 = h * func(t + ( 0  )*h, x + (    3/205 )*k1 + ( 0  )*k2 + (   0  )*k3 + (    0   )*k4 + (    0    )*k5 + (  -6/41 )*k6 + (  -3/205 )*k7 + (-3/41 )*k8 + ( 3/41 )*k9 + ( 6/41 )*k10 + (  0    )*k11)
#         k13 = h * func(t + ( 1  )*h, x + (-1777/4100)*k1 + ( 0  )*k2 + (   0  )*k3 + (-341/164)*k4 + (4496/1025)*k5 + (-289/82 )*k6 + (2193/4100)*k7 + (51/82 )*k8 + (33/164)*k9 + (19/41 )*k10 + (  0    )*k11 + (  1   )*k12)
#         # x12 =                        x + (   41/840 )*k1 + ( 0  )*k2 + (   0  )*k3 + (    0   )*k4 + (    0    )*k5 + (  34/105)*k6 + (   9/35  )*k7 + ( 9/35 )*k8 + ( 9/280)*k9 + ( 9/280)*k10 + (41/840)*k11 + (  0   )*k12 + (  0   )*k13
#         # x13 =                        x + (     0    )*k1 + ( 0  )*k2 + (   0  )*k3 + (    0   )*k4 + (    0    )*k5 + (  34/105)*k6 + (   9/35  )*k7 + ( 9/35 )*k8 + ( 9/280)*k9 + ( 9/280)*k10 + (  0   )*k11 + (41/840)*k12 + (41/840)*k13
#         # from Baselisk docs
#         x12 =                        x + (  -41/840 )*k1 + ( 0  )*k2 + (   0  )*k3 + (    0   )*k4 + (    0    )*k5 + (  34/105)*k6 + (   0     )*k7 + ( 0    )*k8 + ( 0    )*k9 + ( 0    )*k10 + (-41/840)*k11 + (41/840)*k12 + (41/840)*k13
#         x13 =                        x + (     0    )*k1 + ( 0  )*k2 + (   0  )*k3 + (    0   )*k4 + (    0    )*k5 + (  34/105)*k6 + (   9/35  )*k7 + ( 9/35 )*k8 + ( 9/280)*k9 + ( 9/280)*k10 + (  0    )*k11 + (41/840)*k12 + (41/840)*k13
#         # fmt: on

#         # error = norm(-41 / 840 * (k1 + k11 - k12 - k13))
#         error = norm(x12 - x13)
#         print(round(t, 4), h, log10(error))

#         # no div0
#         s = 0.9 * (tol / error) ** 0.2 if error != 0 else 2

#         if s > 10:
#             s = 10
#         if s < 0.1:
#             s = 0.1

#         # error=0
#         if error <= tol:
#             t += h
#             x = x13
#             ts = concatenate((ts, array([t], dtype=float64)))
#             xs = concatenate((xs, expand_dims(x, axis=0)))

#         h *= s

#     return ts, xs


# shamelessly stolen from scipy and adapted
@njit(cache=True)
def dop853(
    func: Callable,
    t_span: Tuple[float, float],
    x0: NDArray,
    atol: float = 1e-10,
    rtol: float = 1e-10,
    init_step: float = 1.0,
    args: Tuple = (),
) -> Tuple[NDArray, NDArray]:
    """High order adaptive RK method

    Args:
        func (Callable): dynamics function
        t_span (Tuple[float, float]): beginning and end times
        x0 (NDArray): initial state
        atol (float, optional): absolute tolerence. Defaults to 1e-10.
        rtol (float, optional): rel tolerence. Defaults to 1e-10.
        init_step (float, optional): initial step size. Defaults to 1.0.
        args (Tuple, optional): additional args to func(t, x, *args). Defaults to ().

    Returns:
        Tuple[NDArray, NDArray]: ts (N, ), xs (nx, N)
    """
    n = len(x0)

    K = np.empty((coefs.n_stages + 1, n), dtype=np.float64)

    t0, tf = t_span
    t = t0
    xs = expand_dims(x0, axis=0)
    ts = array([t0], dtype=float64)
    x = x0
    h = init_step

    # pp180 of RKEM
    while t < tf:
        if t + h > tf:
            h = tf - t

        # STEP
        K[0] = func(t, x)
        for sm1 in range(coefs.N_STAGES - 1):
            s = sm1 + 1
            a = coefs.A[s]
            c = coefs.C[s]
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = func(t + c * h, x + dy, *args)

        xnew = x + h * np.dot(K[:-1].T, coefs.B)

        K[-1] = func(t + h, xnew, *args)

        # END STEP

        # error estimator:
        scale = atol + np.maximum(np.abs(x), np.abs(xnew)) * rtol
        err5 = np.dot(K.T, coefs.E5) / scale
        err3 = np.dot(K.T, coefs.E3) / scale
        err5_norm_2 = np.linalg.norm(err5) ** 2
        err3_norm_2 = np.linalg.norm(err3) ** 2
        denom = err5_norm_2 + 0.01 * err3_norm_2
        error = np.abs(h) * err5_norm_2 / np.sqrt(denom * len(scale))
        # END ERROR ESTIMATOR

        hscale = 0.9 * error ** (-1 / 8) if error != 0 else 2

        if hscale > 10:
            hscale = 10
        if hscale < 0.1:
            hscale = 0.1

        # error=0
        if error <= 1:
            t += h
            x = xnew
            ts = concatenate((ts, array([t], dtype=float64)))
            xs = concatenate((xs, expand_dims(x, axis=0)))

        h *= hscale

    # interpolate as needed
    # if t_eval is not None:
    #     t_eval = np.array(t_eval)
    #     xs =

    return ts, xs.T
