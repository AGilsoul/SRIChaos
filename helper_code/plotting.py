def return_map(
    x,
    y,
    title=None,
    xlabel=None,
    ylabel=None,
    savefig=None,
):
    """
    Plots y against x, overlaying the identity.

    Parameters
    ----------
    x : 1D array-like of float
        A vector indicating the domain of the system.
    y : 1D array-like of float
        A vector indicating the first iterate of the system.
    title : str, optional
        The plot's title. Default is `None`.
    xlabel : str, optional
        The x-axis label. Default is `'$x$'`.
    ylabel : str, optional
        The y-axis label. Default is `'$f(x)$'`.
    savefig : str, optional
        If desired, save the figure at this path (including filename and
        extension).

    Returns
    -------
    None
    """
    # Plot a cobweb diagram with no individual orbit.
    return cobweb(x, y, None, title, xlabel, ylabel, savefig)


def cobweb(
    x,
    y,
    orbit,
    title=None,
    xlabel="$x$",
    ylabel="$f(x)$",
    savefig=None,
):
    """
    Plots the cobweb diagram for an `orbit` on a return map given by vector `x`
    and its iterates `y`.

    Parameters
    ----------
    x : 1D array-like of float
        A vector indicating the domain of the system.
    y : 1D array-like of float
        A vector indicating the first iterate of the system.
    orbit : 1D array-like of float
        A single orbit to plot as a cobweb, including the initial condition.
    title : str, optional
        The plot's title. Default is `None`.
    xlabel : str, optional
        The x-axis label. Default is `'$x$'`.
    ylabel : str, optional
        The y-axis label. Default is `'$f(x)$'`.
    savefig : str, optional
        If desired, save the figure at this path (including filename and
        extension).

    Returns
    -------
    None
    """
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()

    # Axis and label formatting
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(x), max(x))

    # If given an actual orbit, plot the cobweb.
    try:
        N = len(orbit)
        if N > 1:
            ax.vlines(orbit[0], 0, orbit[0])
            for i in range(N - 1):
                ax.vlines(orbit[i], orbit[i], orbit[i + 1])
                ax.hlines(orbit[i + 1], orbit[i], orbit[i + 1])

    # Plot the map and identity on top.
    except:
        pass
    finally:
        ax.plot(x, x, linestyle="dashed", color="grey")
        ax.plot(x, y, color="black")

    if isinstance(savefig, str):
        plt.savefig(savefig)

    plt.show()
