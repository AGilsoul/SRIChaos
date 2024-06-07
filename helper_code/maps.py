def logistic_map(r, x, N=1):
    """
    Evolves the logistic map with growth rate `r` from initial condition(s) `x`
    for `N` steps.

    Parameters
    ----------
    r : float
        The growth rate of the map.
    x : float or 1D array-like
        Either a single initial condition, or a set of initial conditions.
    N : int, optional
        The number of iterations. Default is a single step.

    Returns
    -------
    float or numpy.array of float
        In case `x` is a float and `N == 1`, simply returns the next iterate.
        In case `x` is a 1D array of initial conditions and `N == 1`, returns a
        1D array of next iterates for each initial condition.
        In case `x` is a 1D array of initial conditions and `N > 1`, returns an
        array with shape `(len(x), N+1)`, where each row corresponds to an
        initial condition and columns indicate successive iterations.

    """
    from numpy import asarray, empty

    X = asarray(x)

    if N == 1:
        Y = r * X * (1 - X)
    else:
        Y = empty(X.shape + (N + 1, ))
        try:
            Y[:, 0] = X
            for i in range(N):
                Y[:, i + 1] = r * Y[:, i] * (1 - Y[:, i])
        except:
            Y[0] = X
            for i in range(N):
                Y[i + 1] = r * Y[i] * (1 - Y[i])
    return Y


def tent_map(a, x, N=1):
    """
    Evolves the tent map with slope `a` starting from `x` for `N` iterations.

    Parameters
    ----------
    a : float
        The growth rate of the map.
    x : float or 1D array-like
        Either a single initial condition, or a set of initial conditions.
    N : int, optional
        The number of iterations. Default is a single step.

    Returns
    -------
    float or numpy.array of float
        In case `x` is a float and `N == 1`, simply returns the next iterate.
        In case `x` is a 1D array of initial conditions and `N == 1`, returns a
        1D array of next iterates for each initial condition.
        In case `x` is a 1D array of initial conditions and `N > 1`, returns an
        array with shape `(len(x), N+1)`, where each row corresponds to an
        initial condition and columns indicate successive iterations (the first
        column is of the initial conditions).
    """
    from numpy import asarray, empty, minimum

    X = asarray(x)

    if N == 1:
        Y = a * minimum(X, 1 - X)
    else:
        Y = empty(X.shape + (N + 1, ))
        try:
            # If we got more than one initial conditions...
            Y[:, 0] = X
            for i in range(N):
                Y[:, i + 1] = a * minimum(Y[:, i], 1 - Y[:, i])
        except:
            # Otherwise...
            Y[0] = X
            for i in range(N):
                Y[i + 1] = a * minimum(Y[i], 1 - Y[i])
    return Y
