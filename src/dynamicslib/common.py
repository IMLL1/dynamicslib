import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from typing import Callable, List
from scipy.linalg import null_space
from scipy.integrate import solve_ivp
from tqdm import tqdm
from numba import njit

from dynamicslib.consts import muEM

# %% generic CR3BP stuff
@njit(cache=True)
def get_L1(mu=muEM, tol=1e-14):
    # find x_L1
    x = 1 - 2 * mu
    f = np.inf
    while abs(f) > tol:
        f = -(1 - mu) / (x + mu) ** 2 + mu / (x - 1 + mu) ** 2 + x
        df = 2 * (1 - mu) / (x + mu) ** 3 - 2 * mu / (x - 1 + mu) ** 3 + 1
        dx = -f / df
        x += dx
    return x


@njit(cache=True)
def get_L2(mu=muEM, tol=1e-14):
    # find x_L2
    x = 1 + mu
    f = np.inf
    while abs(f) > tol:
        f = -(1 - mu) / (x + mu) ** 2 - mu / (x - 1 + mu) ** 2 + x
        df = 2 * (1 - mu) / (x + mu) ** 3 + 2 * mu / (x - 1 + mu) ** 3 + 1
        dx = -f / df
        x += dx
    return x


@njit(cache=True)
def get_L3(mu=muEM, tol=1e-14):
    # find x_L3
    x = -1 - mu
    f = np.inf
    while abs(f) > tol:
        f = (1 - mu) / (x + mu) ** 2 + mu / (x - 1 + mu) ** 2 + x
        df = -2 * (1 - mu) / (x + mu) ** 3 - 2 * mu / (x - 1 + mu) ** 3 + 1
        dx = -f / df
        x += dx
    return x


def get_Lpts(mu: float = muEM):
    lagrange_points = np.array(
        [
            [get_L1(mu), get_L2(mu), get_L3(mu), mu - 1 / 2, mu - 1 / 2],
            [0, 0, 0, -np.sqrt(3) / 2, np.sqrt(3) / 2],
        ],
    )
    return lagrange_points


@njit(cache=True)
def U_hess(pos: NDArray[np.float64], mu: float = muEM) -> NDArray[np.float64]:
    x, y, z = pos
    r1mag = np.sqrt(y**2 + z**2 + (mu + x) ** 2)
    r2mag = np.sqrt(y**2 + z**2 + (mu + x - 1) ** 2)
    r1 = pos - np.array([-mu, 0, 0])
    r2 = pos - np.array([1 - mu, 0, 0])
    r1mag = np.sqrt(np.sum(r1**2))
    r2mag = np.sqrt(np.sum(r2**2))
    H = (
        np.diag(np.array([1, 1, 0]))
        + 3 * (1 - mu) / r1mag**5 * np.outer(r1, r1)
        - (1 - mu) / r1mag**3 * np.eye(3)
        + 3 * mu / r2mag**5 * np.outer(r2, r2)
        - mu / r2mag**3 * np.eye(3)
    )

    return H


@njit(cache=True)
def get_A(state: NDArray[np.float64], mu: float = muEM) -> NDArray[np.float64]:
    pos = state[:3]
    Uxx = U_hess(pos, mu)
    O = np.zeros((3, 3))
    I = np.eye(3)
    Omega = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])
    A1 = np.concatenate((O, I), axis=1)
    A2 = np.concatenate((Uxx, Omega), axis=1)
    A = np.concatenate((A1, A2), axis=0)
    return A


@njit(cache=True)
def eom_jac(_, state: NDArray[np.float64], mu: float = muEM) -> NDArray[np.float64]:
    return get_A(state, mu)


@njit(cache=True)
def eom(_, state: NDArray[np.float64], mu: float = muEM) -> NDArray[np.float64]:
    x, y, z, vx, vy, vz = state[:6]
    xyz = state[:3]
    r1vec = xyz + np.array([mu, 0, 0])
    r2vec = xyz + np.array([mu - 1, 0, 0])
    r1mag = np.linalg.norm(r1vec)
    r2mag = np.linalg.norm(r2vec)

    ddxyz = (
        -(1 - mu) * r1vec / r1mag**3
        - mu * r2vec / r2mag**3
        + np.array([2 * vy + x, -2 * vx + y, 0])
    )

    dstate = np.zeros(6)
    dstate[:3] = state[3:]
    dstate[3:] = ddxyz
    return dstate


@njit(cache=True)
def coupled_stm_eom(
    _, state: NDArray[np.float64], mu: float = muEM
) -> NDArray[np.float64]:
    pv = state[:6]
    dpv = eom(None, pv, mu)
    stm = state[6:].reshape((6, 6))
    A = get_A(pv, mu)  # pv[:3]
    dstm = A @ stm

    dstate = np.array([*dpv, *dstm.flatten()])
    return dstate


@njit(cache=True)
def jacobi_constant(state: NDArray[np.float64], mu: float = muEM) -> float:
    x, y, z = state[:3]
    r1mag = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2mag = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    Ugrav = -((1 - mu) / r1mag + mu / r2mag)
    Ucent = -0.5 * (x**2 + y**2 + z**2)
    U = Ucent + Ugrav
    JC = -2 * U
    JC -= sum(state[3:] ** 2)
    return JC


def get_stab(eigval: float, eps: float = 1e-5) -> int:
    """Get stability modes of a single eigenvalue. Numeric codes are
    ```
    0: parabolic
    1: elliptic
    2: +hyperbolic
    3: -hyperbolic
    4: quadrouple
    ```

    Args:
        eigval (float): eigenvalue
        eps (float, optional): epsilon for ==1. Defaults to 1e-5.

    Returns:
        int: the stability type.
    """
    if 1 - eps <= np.abs(eigval) <= eps:
        if np.abs(np.imag(eigval)) < eps:
            return 0
        else:
            return 1
    elif np.abs(np.imag(eigval)) < eps:
        if np.real(eigval) > 0:
            return 2
        else:
            return 3
    else:
        return 4
    
    
# shortcut to get x,y,z from X
def prop_ic(X: NDArray, X2xtf_func: Callable, mu: float = muEM, int_tol=1e-12):
    x0, tf = X2xtf_func(X)

    odesol = solve_ivp(
        eom,
        (0, tf),
        x0,
        rtol=int_tol,
        atol=int_tol,
        method="LSODA",
        jac=eom_jac,
        args=(mu,),
    )
    x, y, z = odesol.y[:3]
    return x, y, z


# shortcut to get JC and tf from X
def get_JC_tf(X: NDArray, X2xtf_func: Callable, mu: float = muEM):
    x0, tf = X2xtf_func(X)
    jc = jacobi_constant(x0, mu)

    return jc, tf
