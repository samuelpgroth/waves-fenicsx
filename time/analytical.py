# Analytical solution for scattering of a plane wave by a sound-hard circle,
# i.e., with the Neumann data set to zero on the circle boundary.
# Samuel Groth
# Cambridge, 20/11/19


def sound_hard_circle(k, rad, plot_grid):
    # from pylab import find
    from scipy.special import jv, hankel1
    import numpy as np
    x = np.vstack((plot_grid[0].ravel(), plot_grid[1].ravel()))
    points = x
    fem_xx = points[0, :]
    fem_xy = points[1, :]
    r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
    theta = np.arctan2(fem_xy, fem_xx)
    npts = np.size(fem_xx, 0)
    a = rad

    n_terms = np.int(30 + (k * a)**1.01)
    k0 = k

    Nx = plot_grid.shape[1]
    Ny = plot_grid.shape[2]

    u_inc = np.exp(1j * k0 * fem_xx)
    n_int = np.where(r < a)
    u_inc[n_int] = 0.0
    u_plot = u_inc.reshape(Nx, Ny)

    u_sc = np.zeros((npts), dtype=np.complex128)
    for n in range(-n_terms, n_terms):
        bessel_deriv = jv(n-1, k0*a) - n/(k0*a) * jv(n, k0*a)
        hankel_deriv = n/(k0*a)*hankel1(n, k0*a) - hankel1(n+1, k0*a)
        u_sc += -(1j)**(n) * (bessel_deriv/hankel_deriv) * hankel1(n, k0*r) * \
            np.exp(1j*n*theta)

    u_sc[n_int] = 0.0
    u_scat = u_sc.reshape(Nx, Ny)
    u_tot = u_scat + u_plot

    return u_tot


def sound_soft_circle(k, rad, plot_grid):
    # from pylab import find
    from scipy.special import jv, hankel1
    import numpy as np
    x = np.vstack((plot_grid[0].ravel(), plot_grid[1].ravel()))
    points = x
    fem_xx = points[0, :]
    fem_xy = points[1, :]
    r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
    theta = np.arctan2(fem_xy, fem_xx)
    npts = np.size(fem_xx, 0)
    a = rad

    n_terms = np.int(30 + (k * a)**1.01)
    k0 = k

    Nx = plot_grid.shape[1]
    Ny = plot_grid.shape[2]

    u_inc = np.exp(1j * k0 * fem_xx)
    n_int = np.where(r < a)
    u_inc[n_int] = 0.0
    u_plot = u_inc.reshape(Nx, Ny)

    u_sc = np.zeros((npts), dtype=np.complex128)
    for n in range(-n_terms, n_terms):
        u_sc += -(1j)**(n) * (jv(n, k0*a)/hankel1(n, k0*a)) * \
                hankel1(n, k0*r) * np.exp(1j*n*theta)

    u_sc[n_int] = 0.0
    u_scat = u_sc.reshape(Nx, Ny)
    u_tot = u_scat + u_plot

    return u_tot


def penetrable_circle(k0, k1, rad, plot_grid):
    # from pylab import find
    from scipy.special import jv, hankel1
    import numpy as np
    x = np.vstack((plot_grid[0].ravel(), plot_grid[1].ravel()))
    points = x
    fem_xx = points[0, :]
    fem_xy = points[1, :]
    r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
    theta = np.arctan2(fem_xy, fem_xx)
    npts = np.size(fem_xx, 0)
    a = rad

    n_terms = np.max([100, np.int(55 + (k0 * a)**1.01)])

    Nx = plot_grid.shape[1]
    Ny = plot_grid.shape[2]

    u_inc = np.exp(1j * k0 * fem_xx)
    n_int = np.where(r < a)
    n_ext = np.where(r >= a)
    u_inc[n_int] = 0.0
    u_plot = u_inc.reshape(Nx, Ny)

    u_int = np.zeros(npts, dtype=np.complex128)
    u_ext = np.zeros(npts, dtype=np.complex128)
    for n in range(-n_terms, n_terms):
        bessel_k0 = jv(n, k0 * rad)
        bessel_k1 = jv(n, k1 * rad)

        hankel_k0 = hankel1(n, k0 * rad)

        bessel_deriv_k0 = jv(n-1, k0 * rad) - n/(k0 * rad) * jv(n, k0 * rad)
        bessel_deriv_k1 = jv(n-1, k1 * rad) - n/(k1 * rad) * jv(n, k1 * rad)

        hankel_deriv_k0 = n/(k0 * rad) * hankel_k0 - hankel1(n+1, k0 * rad)

        a_n = (1j**n) * (k1 * bessel_deriv_k1 * bessel_k0 -
                         k0 * bessel_k1 * bessel_deriv_k0) / \
                        (k0 * hankel_deriv_k0 * bessel_k1 -
                         k1 * bessel_deriv_k1 * hankel_k0)
        b_n = (a_n * hankel_k0 + (1j**n) * bessel_k0) / bessel_k1

        u_ext += a_n * hankel1(n, k0 * r) * np.exp(1j * n * theta)
        u_int += b_n * jv(n, k1 * r) * np.exp(1j * n * theta)
    
    u_int[n_ext] = 0.0
    u_ext[n_int] = 0.0
    u_sc = u_int + u_ext
    u_scat = u_sc.reshape(Nx, Ny)
    u_tot = u_scat + u_plot

    return u_tot
