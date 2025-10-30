from dynamicslib.targeter import *


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
        try:
            X, dF, stm = dc_arclen(X, tangent, f_df_stm_func, s, tol, modified)
        except np.linalg.LinAlgError as err:
            print(f"Linear algebra error encountered: {err}")
            print("returning what's been calculated so far")
            break

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
def natural_param_cont(
    X0: NDArray,
    f_df_stm_func: Callable[
        [float], Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]]
    ],
    param0: float = 0,
    dparam: float = 1e-2,
    N: int = 10,
    tol: float = 1e-10,
    stop_callback: Callable | None = None,
    stop_kwags: dict = {},
) -> Tuple[List, List, List]:
    """Natural parameter continuation continuation wrapper.

    Args:
        X0 (NDArray): initial control variables
        f_df_stm_func (Callable): function with signature f, df/dX, STM = f_df_func(X, cont_parameter)
        param0 (float): the initial value of the parameter.
        dparam (float): The step in natural parameter to take each iteration
        N (int): The number of steps after which to terminate
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

    param = param0
    params = [param0]

    _, dF, stm = f_df_stm_func(param)(X0)

    Xs = [X0]
    eig_vals = [np.linalg.eigvals(stm)]

    bar = tqdm(total=N)
    i = 0
    # ensure that the stopping condition hasnt been satisfied
    while i < N and not (param > param0 and stopfunc(X, eig_vals[-1], eig_vals[-2])):
        X, dF, stm = dc_npc(X + dparam, f_df_stm_func(param), tol)
        params.append(param)
        Xs.append(X)
        eig_vals.append(np.linalg.eigvals(stm))
        param += dparam
        bar.update(1)
        i += 1

    bar.close()

    return Xs, eig_vals, params


def find_bifurc(
    X0: NDArray,
    f_df_stm_func: Callable[[NDArray], Tuple[NDArray, NDArray, NDArray]],
    dir0: NDArray | List,
    s: float = 1e-3,
    tol: float = 1e-10,
    skip_changes: int = 0,
    stabEps: float = 1e-5,
) -> Tuple[NDArray, NDArray]:
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
        NDArray: Bifurcation control variables, tangent vector
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
            if abs(svd.S[-2]) <= 0.5:
                # if svd.
                if skip_changes == 0:
                    tangent = svd.Vh[-2]
                    print(f"BIFURCATING @ X={X} in the direction of {tangent}")
                    return X, tangent
                else:
                    skip_changes -= 1
            else:
                pass

        stabs_prev = stabs
