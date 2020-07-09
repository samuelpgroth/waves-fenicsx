#
# Nonlinear field generated by a planar piston in 2D
# ==================================================
#
# In this demo, we:
# * solve the Westervelt equation over a rectangular domain
# * use 1st order absorbing boundary condition to mimic Sommerfeld condition
# * use an explicit 4th-order Runge-Kutta time-stepping scheme
# * employ mass lumping to diagonalise the mass matrix
# * decompose the field into its harmonic components

import numpy as np
from petsc4py import PETSc

import dolfinx
import ufl
from dolfinx import Function, FunctionSpace, RectangleMesh
from dolfinx.mesh import locate_entities_boundary
from ufl import FiniteElement, TestFunction, dx, grad, inner

from mpi4py import MPI
import time
import matplotlib
from matplotlib import pyplot as plt

# import RK
import RK_eval
from dolfinx.cpp.mesh import CellType


class WesterveltEquation:
    def __init__(self, mesh, k, omega, c, wx, wy, amp):
        P = FiniteElement("Lagrange", mesh.ufl_cell(), k)
        self.V = FunctionSpace(mesh, P)
        self.u, self.v = Function(self.V), Function(self.V)
        self.g = Function(self.V)
        self.g_deriv = Function(self.V)
        self.omega = omega
        self.c = c
        self.a = amp

        # GEOMETRY: rectangular domain with 5mm radius piston on left wall
        #  |------------------------------------------------------------|
        #  |                                                            |
        #  |                                                            |
        #  ||                                                           |
        #  ||                                                           |
        #  || <-- 5mm radius piston                                     |
        #  ||                                                           |
        #  ||                                                           |
        #  |                                                            |
        #  |                                                            |
        #  |----------------------------------------------------------- |
        # Locate boundary facets
        tdim = mesh.topology.dim
        # facets0 belong to 5mm radius piston at x=0
        facets0 = locate_entities_boundary(mesh, tdim - 1,
                                           lambda x: (x[0] < 1.0e-6) *
                                           (np.abs(x[1]) < 5e-3))
        # facets1 belong to right hand wall, at x=wx
        facets1 = locate_entities_boundary(mesh, tdim - 1,
                                           lambda x: x[0] > (wx - 1.0e-6))
        # facets2 belong to top and bottom walls, at y=+wy/2 and y=-wy/2
        facets2 = locate_entities_boundary(mesh, tdim - 1,
                                           lambda x:
                                           np.abs(x[1]) > (wy / 2 - 1.0e-6))

        indices, pos = np.unique(np.hstack((facets0, facets1, facets2)),
                                 return_index=True)
        values = np.hstack((np.full(facets0.shape, 1, np.intc),
                            np.full(facets1.shape, 2, np.intc),
                            np.full(facets2.shape, 3, np.intc)))
        marker = dolfinx.mesh.MeshTags(mesh, tdim - 1, indices, values[pos])
        ds = ufl.Measure('ds', subdomain_data=marker, domain=mesh)

        # dv, p = TrialFunction(self.V), TestFunction(self.V)
        p = TestFunction(self.V)
        beta = 3.5
        delta0 = 3.0e-6
        delta = delta0

        # # EXPERIMENT: add absorbing layers to top and bottom ("PMLs")
        # def delta_fun(x):
        #     lam = 2 * np.pi / k
        #     depth = 2 * lam
        #     dist = np.abs(x[1]) - (wy/2 - depth)
        #     inside = (dist >= 0)
        #     # outside = (dist < 0)
        #     return delta0 + inside * (dist**2) * 1e3

        # delta = Function(self.V)
        # delta.interpolate(delta_fun)
        # delta.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
        #                          mode=PETSc.ScatterMode.FORWARD)

        # with XDMFFile(mesh.mpi_comm(), "delta.xdmf", "w",
        #               encoding=XDMFFile.Encoding.HDF5) as file:
        #     file.write_mesh(mesh)
        #     file.write_function(delta)

        # Westervelt equation
        self.L1 = - inner(grad(self.u), grad(p)) * dx \
            - (delta / c**2) * inner(grad(self.v), grad(p)) * dx \
            + (2 * beta / (rho * c**4)) * inner(self.v * self.v, p) * dx \
            - (1 / c) * inner(self.v, p) * ds \
            + inner(self.g, p) * ds(1) \
            + (delta / c**2) * inner(self.g_deriv, p) * ds(1)

        self.lumped = True
        # Vector to be re-used for assembly
        self.b = None

        # TODO: precompile/pre-process Form L

        if self.lumped:
            # Westervelt equation
            a = (1 / c**2) * \
                (1 - 2 * beta * self.u / (rho * c**2)) * p * dx(degree=k) \
                + (delta / c**3) * p * ds

            self.M = dolfinx.fem.assemble_vector(a)
            self.M.ghostUpdate(addv=PETSc.InsertMode.ADD,
                               mode=PETSc.ScatterMode.REVERSE)
        else:
            # TODO: non-lumped version of Westervelt
            # a = (1 / self.c**2) * inner(dv, p) * dx(degree=k * k)
            M = dolfinx.fem.assemble_matrix(a)
            M.assemble()
            self.solver = PETSc.KSP().create(mesh.mpi_comm())
            opts = PETSc.Options()
            opts["ksp_type"] = "cg"
            opts["ksp_rtol"] = 1.0e-8
            self.solver.setFromOptions()
            self.solver.setOperators(M)

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        with u.vector.localForm() as _u, v.vector.localForm() as _v:
            _u.set(0.0)
            _v.set(0.0)
        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec,
           result: PETSc.Vec) -> PETSc.Vec:
        """For du/dt = f(t, u, v), return f"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec,
           result: PETSc.Vec) -> PETSc.Vec:
        """For dv/dt = f(t, u, v), return f"""

        # Plane wave
        with self.g.vector.localForm() as g_local:
            g_local.set(-2 * self.omega / self.c * self.a *
                        np.cos(self.omega * t))

        with self.g_deriv.vector.localForm() as g_deriv_local:
            g_deriv_local.set(2 * self.omega**2 / self.c * self.a *
                              np.sin(self.omega * t))

        # Update fields that f depends on
        u.copy(result=self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                  mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v.vector)
        self.v.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                  mode=PETSc.ScatterMode.FORWARD)

        # Assemble
        if self.b is None:
            self.b = dolfinx.fem.assemble_vector(self.L1)
        else:
            with self.b.localForm() as b_local:
                b_local.set(0.0)
            dolfinx.fem.assemble_vector(self.b, self.L1)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        # Solve
        if result is None:
            result = v.duplicate()
        if self.lumped:
            result.pointwiseDivide(self.b, self.M)
        else:
            self.solver.solve(self.b, result)

        return result


'''                          Problem parameters                             '''
# * frequency f0
# * wavespeed c0
# * density rho
# * amplitude amp
# * rectangle length wx
# * rectangle width wy
# * total simulation time T (long enough for wave to traverse domain + extra)
f0 = 1.0e6
c0 = 1487.0
rho = 998.0
amp = 1e6  # amplitude
wx = 0.06
wy = 0.06
T = wx / c0 + 2.0 / f0
omega = 2 * np.pi * f0

degree = 2   # discretisation degree
nPerLam = 6  # no. elements per wavelength
CFL = 0.9    # CFL constant

lam = 2 * np.pi * c0 / omega  # wavelength
h = lam / nPerLam             # element dimension
period = 1.0 / f0             # temporal period
nx = np.int(np.round(wx / h))
ny = np.int(np.round(wy / h))

# Build mesh
mesh = RectangleMesh(MPI.COMM_WORLD,
                     [np.array([0, -wy/2, 0]), np.array([wx, wy/2, 0])],
                     [nx, ny],
                     CellType.quadrilateral, dolfinx.cpp.mesh.GhostMode.none)


# Determine timestep for Runge-Kutta via CFL
hmin = min(MPI.COMM_WORLD.allgather(mesh.hmin()))
dt = CFL * hmin / (c0 * (2 * degree + 1))  # CFL condition
# Adjust dt so that it exactly divides one period
n_period = np.int(np.ceil(period / dt))
dt = period / n_period
num_steps = np.int(np.ceil(T / dt))

if (MPI.COMM_WORLD.rank == 0):
    print("#time steps = ", num_steps)

# Create model
eqn = WesterveltEquation(mesh, k=degree, omega=omega, c=c0, wx=wx, wy=wy,
                         amp=amp)

# Plot the field on the central axis
n_line = np.int(np.ceil(wx/lam * 20))
x_line = np.linspace(0, wx, n_line)
points = np.vstack((x_line, np.zeros(n_line), np.zeros(n_line)))

'''                             Solve                                       '''
start = time.time()
# u = RK.solve(eqn.f0, eqn.f1, *eqn.init(), dt=dt, num_steps=num_steps)

# RK timestepping with field evaluation at the locations 'points' over the
# final temporal period of simulation (required for harmonic analysis)
u, u_on_axis = RK_eval.solve(eqn.f0, eqn.f1, *eqn.init(), dt=dt,
                             num_steps=num_steps, points=points,
                             n_period=n_period, mesh=mesh)
end = time.time()

if (MPI.COMM_WORLD.rank == 0):
    print('Solve time = ', end - start)


'''                         Harmonic analysis                               '''
# Save the field along the x-axis for harmonic analysis
np.save('harmonics', u_on_axis)

# Perform harmonic analysis along x-axis
N = u_on_axis.shape[1]
u_fft = np.fft.fft(u_on_axis)

# Arrange harmonics in decending order
harmonics = -np.sort(-abs(u_fft[:, np.int(np.round(N/2)):]), axis=1)

x_line = np.linspace(0, wx, u_on_axis.shape[0])

if (MPI.COMM_WORLD.rank == 0):
    fig = plt.figure(figsize=(12, 8))
    matplotlib.rcParams.update({'font.size': 22})
    plt.rc('font', family='serif')
    # plt.rc('text', usetex=True)
    ax = fig.gca()
    plt.plot(x_line, harmonics[:, 0] * 2/N / 1e6)
    plt.plot(x_line, harmonics[:, 1] * 2/N / 1e6)
    plt.plot(x_line, harmonics[:, 2] * 2/N / 1e6)
    plt.plot(x_line, harmonics[:, 3] * 2/N / 1e6)
    plt.legend(('First harmonic', 'Second hamonic', 'Third harmonic',
                'Fourth harmonic'), shadow=True, loc=(0.65, 0.72),
               handlelength=1.5, fontsize=20)
    plt.grid(True)
    # plt.ylim([0, 2])
    # plt.xlim([0, 6])
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
    plt.xlabel(r'$x$ (cm)')
    plt.ylabel('Pressure (MPa)')
    fig.savefig('harmonics.png')
