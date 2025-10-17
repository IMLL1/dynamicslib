from typing import List, Callable
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

from dynamicslib.consts import muEM
from dynamicslib.common import get_Lpts


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
    alpha: float = 1,
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
        curves3d.append(plotly_curve(x, y, z, lbl, color=c, width=5,opacity=alpha))
        projs.append(
            plotly_curve(
                x * 0 + projX, y, z, lbl, color=c, width=2, opacity=0.75 * alpha
                
            )
        )
        projs.append(
            plotly_curve(
                x, 0 * y + projY, z, lbl, color=c, width=2, opacity=0.75 * alpha
            )
        )
        projs.append(
            plotly_curve(
                x, y, 0 * z + projZ, lbl, color=c, width=2, opacity=0.75 * alpha
            )
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
        width=1000,
        height=600,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=50),
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
