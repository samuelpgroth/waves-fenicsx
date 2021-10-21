# Analytical solution for scattering of a plane wave by a sound-hard circle,
# i.e., with the Neumann data set to zero on the circle boundary.
# Samuel Groth
# Cambridge, 20/11/19

import scipy
import numpy as np
from scipy.special import jv, hankel1


def sound_hard_circle(k, rad, plot_grid):
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


def sphere_density_contrast(sizeParam, n, Nx, rho1, rho2, plot_grid, a):

    # freq = 1000
    Ny = Nx
    # Ny = 100
    c01 = 1000
    # sizeParam = 5
    # a = 1
    c02 = c01/n

    # Unit amplitude for incident plane wave
    # p_max = 1.

    k1 = sizeParam/a
    k2 = k1*n

    beta = rho1 * c01 / (rho2 * c02)

    # We want to have at least one wavelength
    # rGammaR = a  # a + wl_air
    Nterms = 100
    # dpml = 0

    # rGammaS = a
    # DomainR = rGammaR

    # Hack for VIE comparisons
    # dx = DomainR * 2 / Nx

    # xmin,xmax,ymin,ymax=[-DomainR+dx/2,DomainR-dx/2,
    #                      -DomainR+dx/2,DomainR-dx/2];
    # plot_grid = np.mgrid[xmin:xmax:Nx*1j,ymin:ymax:Ny*1j];

    points = np.vstack((plot_grid[0].ravel(), plot_grid[1].ravel(),
                        np.array([0]*plot_grid[0].size)))

    # plot_me = np.zeros(points.shape[1],dtype=np.complex128)

    x, y, z = points

    # sphere_interior = np.sqrt(points[0, :]**2 + points[1, :]**2 +
    #                           points[2, :]**2)
    # idx_exterior = (sphere_interior >= DomainR-dpml)

    fem_xx = points[0, :]
    fem_xy = points[1, :]
    # colors = np.random.rand(10)
    # plt.plot(fem_xx, fem_xy, 'ob')
    # plt.show()
    # area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
    npts = np.size(fem_xx, 0)

    # set the vector for number of terms:
    m = np.arange(Nterms+1)
    # set up vector for scattered field:
    p_s = np.zeros((npts, 1), dtype=np.complex128)
    # zz = np.zeros((npts, 1))
    # r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy + zz * zz)
    r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
    theta = np.arctan2(fem_xy, fem_xx)

    # print('frequency = ', freq);
    # print('k(air) = ', k1);
    # print('k(water) =', k2);
    # Legendre polynomial terms
    P_m = np.zeros((Nterms+1, npts), dtype=np.complex128)
    for m in range(0, Nterms+1):  # I need to access all (N+1) places in the
        # vector P_m. This is why, the range goes from 0 to N+1
        for j in range(0, len(theta)):
            th = theta[j]
            # the legendre function does not work with vectors, so passing
            # each value from the vector theta:
            aa = scipy.special.lpmn(m, m, np.cos(th))
            P_m[m, j] = aa[0][0, m]

    # print('computing field for transmission problem..')
    for m in range(0, Nterms+1):
        j_m_k1a = scipy.special.spherical_jn(m, k1*a, False)
        y_m_k1a = scipy.special.spherical_yn(m, k1*a, False)
        j_m_k2a = scipy.special.spherical_jn(m, k2*a, False)
        # Derivative of spherical Bessel function
        j_m_k1a_prime = scipy.special.spherical_jn(m, k1*a, True)
        y_m_k1a_prime = scipy.special.spherical_yn(m, k1*a, True)
        j_m_k2a_prime = scipy.special.spherical_jn(m, k2*a, True)
        # Hankel function
        h_m_k1a = complex(j_m_k1a, -y_m_k1a)
        h_m_k1a_prime = complex(j_m_k1a_prime, -y_m_k1a_prime)
        D = (-1.+0.0j)**(1.-(m/2.)) * (2.*m+1.) / \
            (h_m_k1a_prime * j_m_k2a - beta * h_m_k1a * j_m_k2a_prime)
        A = (j_m_k2a * j_m_k1a_prime - beta * j_m_k1a * j_m_k2a_prime) * D
        B = (h_m_k1a * j_m_k1a_prime - h_m_k1a_prime * j_m_k1a) * D

        for ipt in range(0, len(fem_xx)):
            # radial distance from the center of sphere
            radius = np.sqrt(fem_xx[ipt]**2 + fem_xy[ipt]**2)
            if (radius >= a):
                j_m_k1r = scipy.special.spherical_jn(m, k1*radius, False)
                y_m_k1r = scipy.special.spherical_yn(m, k1*radius, False)
                # second kind spherical hankel function:
                h_m_k1r = complex(j_m_k1r, -y_m_k1r)
                p_s[ipt] += A * h_m_k1r * P_m[m, ipt]
            else:
                j_m_k2r = scipy.special.spherical_jn(m, k2*radius, False)
                p_s[ipt] += B * j_m_k2r * P_m[m, ipt]

    # set up incident field everywhere including the the interior of sphere:
    p_i = np.zeros((npts, 1), dtype=np.complex128)
    for j in range(0, npts):
        p_i[j] = np.exp(-1j*k1*fem_xx[j])
    # find the radius of each evaluation point from the center of sphere:
    r = np.sqrt(fem_xx**2 + fem_xy**2)
    # find which points lie in the interior of sphere:
    n_int = np.where(r < a)
    # set the incident field interior to sphere to be zero:
    p_i[n_int] = complex(0.0, 0.0)
    # add the resulting incident field to the scattered field computed before:
    p_t = p_s + p_i
    # p_t[idx_exterior] = 0.

    P = p_t.reshape((Nx, Ny))

    return P
