import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from typing import Callable, List
from scipy.linalg import null_space
from scipy.integrate import solve_ivp
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

muEM = 1.215058560962404e-2
LU = 384400
TU = 406067  # NASA says lunar period is 29.53 days; this is that/2pi in sec, rounded to the sec


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

    G = np.array([np.inf] * nX)
    niters = 0
    dX = np.array([np.inf] * nX)
    while np.linalg.norm(G) > tol and np.linalg.norm(dX) > tol:
        f, dF, stm_full = f_df_func(X)
        dX = -np.linalg.inv(dF) @ f
        X += dX
        niters += 1

    return X, dF, stm_full


def arclen_cont(
    X0: NDArray,
    f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
    dir0: NDArray | List,
    s: float = 1e-3,
    S: float = 0.5,
    tol: float = 1e-10,
    stop_callback: Callable | None = None,  # possibly change to also take dF?
    stop_kwags: dict = {},
    modified: bool = False,
) -> Tuple[List, List]:
    """Pseudoarclength continuation wrapper. The modified algorithm has a full step size of s, rather than projected step size.

    Args:
        X0 (NDArray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        dir0 (NDArray | List): rough initial stepoff direction. Is mostly just used to switch the direction of the computed tangent vector
        s (float, optional): step size. Defaults to 1e-3.
        S (float, optional): terminate at this arclength. Defaults to 0.5.
        tol (float, optional): tolerance for convergence. Defaults to 1e-10.
        stop_callback (Callable): Function with signature f(X, current_eigvals, previous_eigvals, *kwargs) which returns True when continuation should terminate. If None, will only terminate when the final arclength is reached. Defaults to None.
        stop_kwags (dict, optional): keyword arguments to stop_calback. Defaults to {}.
        modified (bool, optional): Whether to use modified algorithm. Defaults to False.

    Returns:
        Tuple[List, List]: all Xs, all eigenvalues
    """
    # if no stop callback, make one
    if callable(stop_callback):
        stopfunc = lambda X, ecurr, elast: stop_callback(X, ecurr, elast, **stop_kwags)
    else:
        stopfunc = lambda X, ecurr, elast: False

    X = X0.copy()
    tangent_prev = dir0

    _, dF, stm = f_df_stm_func(X0)
    svd = np.linalg.svd(dF)
    tangent = svd.Vh[-1]

    # if the direction we asked for is normal to the computed tangent, use the second-most tangent vector
    if np.abs(np.dot(tangent, dir0)) < 1e-5:
        tangent = svd.Vh[-2]

    Xs = [X0]
    eig_vals = [np.linalg.eigvals(stm)]

    bar = tqdm(total=S)
    arclen = 0.0

    # ensure that the stopping condition hasnt been satisfied
    while arclen < S and not (arclen > 0 and stopfunc(X, eig_vals[-1], eig_vals[-2])):
        # if we flip flop, undo the flipflop
        if np.dot(tangent, tangent_prev) < 0:
            tangent *= -1
        X, dF, stm = dc_arclen(X, tangent, f_df_stm_func, s, tol, modified)

        Xs.append(X)

        eig_vals.append(np.linalg.eigvals(stm))
        dS = np.linalg.norm(Xs[-1] - Xs[-2])

        tangent_prev = tangent

        svd = np.linalg.svd(dF)
        tangent = svd.Vh[-1]

        arclen += dS
        bar.update(float(dS))

    bar.close()

    return Xs, eig_vals


# WIP
def param_cont(
    X0: NDArray,
    f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
    step: NDArray,
    S: float = 0.2,
    tol: float = 1e-10,
    stop_callback: Callable | None = None,
    stop_kwags: dict = {},
) -> Tuple[List, List]:
    """Natural parameter continuation continuation wrapper.

    Args:
        X0 (NDArray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        step (NDArray): The step in control variable space to take each iteration
        S (float): The amount of change in control variable space at which to terminate
        tol (float, optional): tolerance for convergence. Defaults to 1e-10.
        stop_callback (Callable): Function with signature f(X, current_eigvals, previous_eigvals, *kwargs) which returns True when continuation should terminate. If None, will only terminate when the final length is reached. Defaults to None.
        stop_kwags (dict, optional): keyword arguments to stop_calback. Defaults to {}.
        modified (bool, optional): Whether to use modified algorithm. Defaults to False.

    Returns:
        Tuple[List, List]: all Xs, all eigenvalues
    """
    # if no stop callback, make one
    if callable(stop_callback):
        stopfunc = lambda X, ecurr, elast: stop_callback(X, ecurr, elast, **stop_kwags)
    else:
        stopfunc = lambda X, ecurr, elast: False

    X = X0.copy()
    dS = np.linalg.norm(step)

    _, dF, stm = f_df_stm_func(X0)

    Xs = [X0]
    eig_vals = [np.linalg.eigvals(stm)]

    bar = tqdm(total=S)
    arclen = 0.0

    # ensure that the stopping condition hasnt been satisfied
    while arclen < S and not (arclen > 0 and stopfunc(X, eig_vals[-1], eig_vals[-2])):
        X, dF, stm = dc_npc(X + step, f_df_stm_func, tol)
        Xs.append(X)
        eig_vals.append(np.linalg.eigvals(stm))
        arclen += dS
        bar.update(float(dS))

    bar.close()

    return Xs, eig_vals


def find_bifurc(
    X0: NDArray,
    f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
    dir0: NDArray | List,
    s: float = 1e-3,
    tol: float = 1e-10,
    skip_changes: int = 0,
    stabEps: float = 1e-5,
) -> NDArray:
    """Find bifurcation using changes in stability. This function can likely be gotten rid of

    Args:
        X0 (NDArray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X)
        dir0 (NDArray | List): rough initial stepoff direction. Is mostly just used to switch the direction of the computed tangent vector
        s (float, optional): step size. Defaults to 1e-3.
        tol (float, optional): tolerance for convergence. Defaults to 1e-10.
        skip_changes (int, optional): number of stability changes to skip. Defaults to 0.
        stabEps (float, optional): Arbitrary epsilon to determine when an eigenvalue = +/-1. Defaults to 1e-5.

    Returns:
        NDArray: Bifurcation control variables
    """
    X = X0.copy()
    tangent_prev = dir0

    _, dF, stm = f_df_stm_func(X0)
    svd = np.linalg.svd(dF)
    tangent = svd.Vh[-1]

    Xs = [X0]

    stabs_prev = [None] * 6

    while True:
        if np.dot(tangent, tangent_prev) < 0:
            tangent *= -1
        X, dF, stm = dc_arclen(X, tangent, f_df_stm_func, s, tol)

        Xs.append(X)

        # eval_norms = np.sort(np.abs(eig_vals[-1]))[3:]
        stabs = sorted([get_stab(e, stabEps) for e in np.linalg.eigvals(stm)])

        tangent_prev = tangent

        # tangent = null_space(dF)
        svd = np.linalg.svd(dF)
        tangent = svd.Vh[-1]

        if stabs != stabs_prev and None not in stabs_prev:
            if skip_changes == 0:
                tangent = svd.Vh[-2]
                print(f"BIFURCATING @ X={X} in the direction of {tangent}")
                return X
            else:
                skip_changes -= 1

        stabs_prev = stabs


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


# %%
def matplotlib_family(
    Xs_all: List,
    xyzs_all: list,
    names: list,
    eig_vals_all: list,
    X2xtf_func: Callable,
    colormap: str = "rainbow",
    mu: float = muEM,
):
    # %config InlineBackend.print_figure_kwargs = {'bbox_inches':None}

    assert len(Xs_all) == len(xyzs_all) == len(names) == len(eig_vals_all)
    Lpoints = get_Lpts(mu)
    num_fams = len(Xs_all)
    cm = plt.get_cmap(colormap)

    for ifam in range(num_fams):
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(121, projection="3d")
        Xs = Xs_all[ifam]
        eig_vals = eig_vals_all[ifam]
        n = len(Xs)
        # plot the 3D
        xyzs_fam = xyzs_all[ifam]
        for i, xyzs in enumerate(xyzs_fam):
            ax.plot(*xyzs, "-", lw=1, color=cm(i / n))

        # find the limits so we can plot and preserve them
        ax.axis("equal")
        lims = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        xl, yl, zl = (
            1.2 * (lims - np.mean(lims, axis=1)[:, None])
            + np.mean(lims, axis=1)[:, None]
        )
        ax.set(xlim=xl, ylim=yl, zlim=zl)

        # plot the projections
        for i, xyzs in enumerate(xyzs_fam):
            x, y, z = xyzs
            ax.plot(xl[0], y, z, "-", lw=0.3, color=cm(i / n), alpha=0.5)
            ax.plot(x, yl[1], z, "-", lw=0.3, color=cm(i / n), alpha=0.5)
            ax.plot(x, y, zl[0], "-", lw=0.3, color=cm(i / n), alpha=0.5)

        # Lpoints and bodies
        ax.scatter(Lpoints[0], Lpoints[1], c="c", s=6, alpha=1, axlim_clip=True)
        ax.scatter([-mu, 1 - mu], [0, 0], c="w", s=25, alpha=1, axlim_clip=True)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo["grid"]["linewidth"] = 0.3
            axis._axinfo["grid"]["color"] = "darkgrey"
            axis.set_label_coords(0, 0)

        ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$", title="Position Space")
        ax.tick_params(axis="both", which="major", labelsize=7)
        fig.suptitle(names[ifam])

        # stability, period, energy
        ax = fig.add_subplot(322)
        ax2 = fig.add_subplot(324, sharex=ax)
        ax3 = fig.add_subplot(326, sharex=ax)
        x_axis = list(range(n))
        periods = []
        jacobi_consts = []
        stabs1 = []
        stabs2 = []
        for i, X in enumerate(Xs):
            jc, tf = get_JC_tf(X, X2xtf_func)
            periods.append(tf)
            jacobi_consts.append(jc)
            eigvals = eig_vals[i]
            eigvals = eigvals[np.argsort(np.abs(eigvals))]
            stabs1.append(np.abs(eigvals[0] + 1 / eigvals[0]) / 2)
            stabs2.append(np.abs(eigvals[1] + 1 / eigvals[1]) / 2)

        ax.scatter(x_axis, jacobi_consts, c=x_axis, alpha=1, cmap=cm)
        ax2.scatter(x_axis, periods, c=x_axis, alpha=1, cmap=cm)
        ax3.scatter(x_axis, stabs1, c=x_axis, alpha=1, cmap=cm)
        ax3.scatter(x_axis, stabs2, c=x_axis, alpha=1, cmap=cm)
        ax.set(ylabel="Jacobi Constant", title="Family Evolution")
        ax2.set(ylabel="Peroid")
        ax3.set(xlabel="Index Along Family", ylabel="Stability Index")
        ax3.set_yscale("log")
        ax.grid(True, lw=0.25)
        ax2.grid(True, lw=0.25)
        ax3.grid(True, lw=0.25)
        ax.set_xticklabels([])
        ax2.set_xticklabels([])
        fig.tight_layout(h_pad=0, w_pad=0)
    plt.show()


# PLOTLY
def plotly_curve(x, y, z, name="", opacity=1.0, **kwargs):
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=kwargs,
        name=name,
        hoverlabel=dict(
            font=dict(size=9), namelength=-1, bgcolor="black", font_color="white"
        ),
        opacity=opacity,
    )


def make_label(data: List | NDArray, param_names: List[str]):
    terms = []
    for val, param in zip(data, param_names):
        valtxt = f"{val}" if param.lower() == "index" else f"{val:.5f}"
        terms.append(f"{param}: {valtxt}")
    return "<br>".join(terms)


def plotly_family(
    xyzs: List,
    name: str,
    data: List | NDArray,
    param_names: list,
    colormap: str = "rainbow",
    mu: float = muEM,
    renderer: str | None = "browser",
    html_save: str | None = None,
):
    data = np.array(data)
    data = data.astype(np.float32)

    assert len(xyzs) == len(data)
    assert len(data[0]) == len(param_names)

    xyzs = [np.float32(xyz) for xyz in xyzs]
    if html_save is not None and html_save.endswith(".html"):
        html_save = html_save.rstrip(".html")

    if renderer is not None:
        pio.renderers.default = renderer

    n = len(xyzs)

    curves = []
    xs, ys, zs = np.hstack(xyzs)
    minx, miny, minz = (min(xs), min(ys), min(zs))
    maxx, maxy, maxz = (max(xs), max(ys), max(zs))
    rng = 1.25 * max([maxx - minx, maxy - miny, maxz - minz])

    ctrX = (maxx + minx) / 2
    ctrY = (maxy + miny) / 2
    ctrZ = (maxz + minz) / 2
    projX = ctrX - rng / 2
    projY = ctrY - rng / 2
    projZ = ctrZ - rng / 2
    projs = []
    curves3d = []
    for i, xyzs in enumerate(xyzs):
        x, y, z = xyzs
        c = px.colors.sample_colorscale(colormap, i / n)[0]
        lbl = make_label(data[i], param_names)
        curves3d.append(plotly_curve(x, y, z, lbl, color=c, width=5))
        projs.append(
            plotly_curve(x * 0 + projX, y, z, lbl, color=c, width=2, opacity=0.75)
        )
        projs.append(
            plotly_curve(x, 0 * y + projY, z, lbl, color=c, width=2, opacity=0.75)
        )
        projs.append(
            plotly_curve(x, y, 0 * z + projZ, lbl, color=c, width=2, opacity=0.75)
        )

    fig = go.Figure(data=[*curves3d, *projs])

    Lpoints = get_Lpts()
    fig.add_trace(
        go.Scatter3d(
            x=Lpoints[0],
            y=Lpoints[1],
            z=0 * Lpoints[0],
            text=[f"L{lp+1}" for lp in range(5)],
            hoverinfo="x+y+text",
            mode="markers",
            marker=dict(color="magenta", size=4),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[-mu, 1 - mu],
            y=[0, 0],
            z=[0, 0],
            mode="markers",
            text=["earth", "moon"],
            hoverinfo="x+y+text",
            marker=dict(color="cyan"),
        )
    )

    fig.update_layout(
        # scene=dict(xaxis_title="x [nd]", yaxis_title="y [nd]", zaxis_title="z [nd]"),
        title=dict(text=name, x=0.5, xanchor="center", yanchor="bottom", y=0.95),
        width=1000,
        height=800,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=0, r=30, b=0, t=50),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
    )
    fig.update_scenes(
        xaxis=dict(
            title="x [nd]",
            showbackground=False,
            showgrid=True,
            zeroline=False,
            range=[ctrX - rng / 2, ctrX + rng / 2],
        ),
        yaxis=dict(
            title="y [nd]",
            showbackground=False,
            showgrid=True,
            zeroline=False,
            range=[ctrY - rng / 2, ctrY + rng / 2],
        ),
        zaxis=dict(
            title="z [nd]",
            showbackground=False,
            showgrid=True,
            zeroline=False,
            range=[ctrZ - rng / 2, ctrZ + rng / 2],
        ),
        aspectmode="cube",
    )

    argshide = {"visible": [*[True] * n, *[False] * (3 * n), True, True]}
    argsshow = {"visible": [*[True] * (4 * n + 2)]}
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.2,
                y=1,
                xanchor="left",
                yanchor="bottom",
                showactive=False,
                buttons=[
                    dict(
                        label="Show<br>Projections", method="restyle", args=[argsshow]
                    ),
                    dict(
                        label="Hide<br>Projections", method="restyle", args=[argshide]
                    ),
                ],
            ),
        ]
    )

    datatr = data.T
    trace = go.Scatter(
        x=list(range(n)),
        y=datatr[0],
        mode="markers",
        name="Parameter Sweep",
        marker=dict(color=px.colors.sample_colorscale(colormap, np.arange(n) / n)),
    )
    fig2 = go.Figure(data=trace)
    fig2.update_layout(
        title=dict(
            text=name + " Parameter Sweep",
            x=0.5,
            xanchor="center",
            yanchor="bottom",
            y=0.95,
        ),
        xaxis=dict(title="Index Along Family"),
        width=700,
        height=400,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=0, r=30, b=0, t=50),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
    )

    # DROPDOWNS
    fig2.update_layout(
        updatemenus=[
            dict(
                buttons=list(
                    [
                        dict(
                            label=param,
                            method="update",
                            args=[{"y": [datatr[i]]}, {"yaxis.title.text": param}],
                        )
                        for i, param in enumerate(param_names)
                    ]
                ),
                direction="down",
                showactive=True,
                x=0,
                xanchor="left",
                y=1,
                yanchor="bottom",
            ),
        ]
    )

    if html_save is not None:
        fig.write_html(html_save + "_3d.html", include_plotlyjs="cdn")
        fig2.write_html(html_save + "_sweep.html", include_plotlyjs="cdn")
    if renderer is not None:
        fig.show()
        fig2.show()
