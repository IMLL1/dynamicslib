from dynamicslib.common import *


# %% continuation
def get_f_df(
    X: NDArray,
    X2xtf: Callable,
    dF_func: Callable,
    f_func: Callable,
    full_period=False,
    mu: float = muEM,
    int_tol: float = 1e-10,
) -> Tuple[NDArray, NDArray, NDArray]:
    x0, tf = X2xtf(X)
    xstmIC = np.array([*x0, *np.eye(6).flatten()])
    ode_sol = solve_ivp(
        coupled_stm_eom,
        (0, tf if full_period else tf / 2),
        xstmIC,
        rtol=int_tol,
        atol=int_tol,
        args=(mu,),
        method="DOP853",
    )

    xf, stm = ode_sol.y[:6, -1], ode_sol.y[6:, -1].reshape(6, 6)
    eomf = eom(0, xf, mu)

    dF = dF_func(eomf, stm)
    f = f_func(x0, tf, xf)

    if not full_period:
        G = np.diag([1, -1, 1, -1, 1, -1])
        Omega = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        I = np.identity(3)
        O = np.zeros((3, 3))
        mtx1 = np.block([[O, -I], [I, -2 * Omega]])
        mtx2 = np.block([[-2 * Omega, I], [-I, O]])
        stm_full = G @ mtx1 @ stm.T @ mtx2 @ G @ stm
    else:
        stm_full = stm
    return f, dF, stm_full


def dc_arclen(
    X_prev: NDArray,
    tangent: NDArray,
    f_df_func: Callable,
    s: float = 1e-3,
    tol: float = 1e-8,
    modified: bool = False,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Pseudoarclength continuation differential corrector. The modified algorithm has a full step size of s, rather than projected step size.

    Args:
        X_prev (NDArray): previous control variables
        tangent (NDArray): tangent to previous orbit. Would be nice to not have to carry over...
        f_df_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        s (float, optional): step size. Defaults to 1e-3.
        tol (float, optional): tolerance for convergence. Defaults to 1e-8.
        modified (boolean, optional): whether to use modified algorithm. Defaults to false.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: X. final dF/dx, full-rev STM
    """

    X = X_prev + s * tangent

    nX = len(X)
    dF = np.empty((nX - 1, nX))
    stm_full = np.empty((nX, nX))

    G = np.array([np.inf] * nX)
    niters = 0
    dX = np.array([np.inf] * nX)
    while np.linalg.norm(G) > tol and np.linalg.norm(dX) > tol:
        f, dF, stm_full = f_df_func(X)
        delta = X - X_prev
        lastG = np.dot(delta, delta) - s**2 if modified else np.dot(delta, tangent) - s
        lastDG = 2 * delta if modified else tangent
        G = np.array([*f, lastG])
        dG = np.vstack((dF, lastDG))
        dX = -np.linalg.inv(dG) @ G
        X += dX
        niters += 1

    return X, dF, stm_full


def dc_npc(
    X_guess: NDArray,
    f_df_func: Callable,
    tol: float = 1e-8,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Natural parameter continuation differetial corrector

    Args:
        X_guess (NDArray): guess for control variables
        f_df_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        tol (float, optional): tolerance for convergence. Defaults to 1e-8.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: X. final dF/dx, full-rev STM
    """

    X = X_guess.copy()

    nX = len(X)
    dF = np.empty((nX, nX))
    stm_full = np.empty((nX, nX))

    f = np.array([np.inf] * nX)
    niters = 0
    dX = np.array([np.inf] * nX)
    while np.linalg.norm(f) > tol and np.linalg.norm(dX) > tol:
        f, dF, stm_full = f_df_func(X)
        dX = -np.linalg.inv(dF) @ f
        X += dX
        niters += 1

    return X, dF, stm_full
