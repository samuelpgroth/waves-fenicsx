#
# Time-harmonic plane wave scattered by a circle
# ==============================================
#
# We solve the linear wave equation in the time domain using Runge-Kutta 4
# to discretise in time.
# The specific problem is a time-harmonic plane wave travelling in the positive
# x-direction being scattered by a circle. We run the simulation long enough
# to achieve a steady state, then the solution can be compared to the
# analytical solution.
# In this demo, we:
# * create a quad mesh using pyGMSH
# * use 1st order absorbing boundary condition to mimic Sommerfeld condition
# * use an explicit 4th-order Runge-Kutta time-stepping scheme
# * employ mass lumping to diagonalise the mass matrix

# NOTE: the length of time required to reach a steady state depends on the
# domain size and the refractive index. Need to experiment.

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import RK_eval_split_field
from dolfinx.io import XDMFFile
import ufl
from dolfinx import Function, FunctionSpace
import matplotlib
from ufl import (TestFunction, TrialFunction, dx, grad, inner, FacetNormal, ds,
                 dot)
import pygmsh
import meshio
from analytical import penetrable_circle

# NOTE: With the current design 'LinearWaveEquation' is
# problem-specific, i.e. it is not for use for all linear wave
# equations. It encodes boundary condition data. Do not attempt to use
# current design as a generic wave solver.


class LinearWaveEquation:
    def __init__(self, mesh, k: int, omega, c, c0, lumped):
        P = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), k)
        self.V = FunctionSpace(mesh, P)
        self.u, self.v = Function(self.V), Function(self.V)
        self.g1 = Function(self.V)
        self.g2 = Function(self.V)
        self.omega = omega
        self.c = c
        self.c0 = c0

        n = FacetNormal(mesh)

        # Pieces for plane wave incident field
        x = ufl.geometry.SpatialCoordinate(mesh)
        cos_wave = ufl.cos(self.omega / self.c0 * x[0])
        sin_wave = ufl.sin(self.omega / self.c0 * x[0])

        plane_wave = self.g1 * cos_wave + self.g2 * sin_wave

        dv, p = TrialFunction(self.V), TestFunction(self.V)
        self.L1 = - inner(grad(self.u), grad(p)) * dx(degree=k) \
            - (1 / self.c) * inner(self.v, p) * ds \
            - (1 / self.c**2) * (-self.omega**2) * inner(plane_wave, p) * dx \
            - inner(grad(plane_wave), grad(p)) * dx \
            + inner(dot(grad(plane_wave), n), p) * ds

        # Vector to be re-used for assembly
        self.b = None

        # TODO: precompile/pre-process Form L

        self.lumped = lumped
        if self.lumped:
            a = (1 / self.c**2) * p * dx(degree=k)
            self.M = dolfinx.fem.assemble_vector(a)
            self.M.ghostUpdate(addv=PETSc.InsertMode.ADD,
                               mode=PETSc.ScatterMode.REVERSE)
        else:
            a = (1 / self.c**2) * inner(dv, p) * dx(degree=k)
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

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) -> PETSc.Vec:
        """For du/dt = f(t, u, v), compute and return f"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) -> PETSc.Vec:
        """For dv/dt = f(t, u, v), compute and return f"""
        # # Update boundary condition
        # with self.g.vector.localForm() as g_local:
        #     g_local.set(-2 * self.omega / self.c * np.cos(self.omega * t))

        # Set up plane wave incident field - it's made of two parts, a cosine
        # and a sine, which are multiplied by cos(omega*t) and sin(omega*t)
        # later on
        with self.g1.vector.localForm() as g1_local:
            g1_local.set(np.cos(self.omega * t))

        with self.g2.vector.localForm() as g2_local:
            g2_local.set(np.sin(self.omega * t))

        # Update fields that f depends on
        u.copy(result=self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                  mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v.vector)
        self.v.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                  mode=PETSc.ScatterMode.FORWARD)
        # Assemble b
        if self.b is None:
            self.b = dolfinx.fem.assemble_vector(self.L1)
        else:
            with self.b.localForm() as b_local:
                b_local.set(0.0)
            dolfinx.fem.assemble_vector(self.b, self.L1)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        if result is None:
            result = v.duplicate()

        # Solve
        if self.lumped:
            result.pointwiseDivide(self.b, self.M)
        else:
            self.solver.solve(self.b, result)

        return result


# Problem parameters
f0 = 6
omega = 2 * np.pi * f0
c0 = 1.0
refInd = 1.1
degree = 2  # discretisation degree
nPerLam = 6  # no. elements per wavelength
T = 1.0  # total simulation time
CFL = 0.48  # CFL constant ( < 1/d for dimension d=2,3 )


class plane_wave:
    """Incident field is a plane wave travelling in +ve x-direction"""
    def __init__(self):
        self.t = 0.0

    def eval(self, x):
        plane_wave = np.exp(1j * omega / c0 * x[0])
        f = np.exp(-1j*omega*self.t) * plane_wave
        return np.real(f)


incident = plane_wave()

lam = 2 * np.pi * c0 / omega  # wavelength
h = lam / nPerLam  # element dimension
period = 1.0 / f0

# Radius of circular scatterer
rad_circ = 1 * lam

wx = rad_circ + 4 * lam
wy = wx
nx = np.int(np.round(wx / h))
ny = np.int(np.round(wy / h))


def circle_mesh():
    ''' Build circular mesh with quad elements '''
    geom = pygmsh.built_in.Geometry()
    disk_inner = geom.add_circle([0, 0, 0], rad_circ, h/refInd)
    disk = geom.add_circle([0, 0, 0], wx/2, h,
                           holes=[disk_inner.line_loop])

    geom.add_raw_code("Recombine Surface {:};")
    geom.add_raw_code("Mesh.RecombinationAlgorithm = 2;")

    mesh = pygmsh.generate_mesh(geom, dim=2, prune_z_0=True)
    cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                if cells.type == "quad"]))
    triangle_mesh = meshio.Mesh(points=mesh.points,
                                cells=[("quad", cells)])

    # Write mesh
    meshio.xdmf.write("mesh.xdmf", triangle_mesh)


circle_mesh()

# Read mesh
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r",
              XDMFFile.Encoding.HDF5) as xdmf:
            mesh = xdmf.read_mesh(name="Grid")


class C_var:
    ''' Wave speed function '''
    def __init__(self):
        self.rad = rad_circ

    def eval(self, x):
        """ Wavespeed function """
        r = np.sqrt((x[0])**2 + (x[1])**2)
        inside = (r <= self.rad)
        outside = (r > self.rad)
        return inside * c0 / refInd + outside * c0


# Function space onto which we interpolate the wavenumber function
V_k = FunctionSpace(mesh, ("Lagrange", 2))
c_temp = C_var()
c = Function(V_k)
c.interpolate(c_temp.eval)

# Determine timestep for Runge-Kutta via CFL
hmin = min(MPI.COMM_WORLD.allgather(mesh.hmin()))
dt = CFL * hmin / (c0 * (2 * degree + 1))  # CFL condition
# Adjust dt so that it exactly divides one period
n_period = np.int(np.ceil(period / dt))
dt = period / n_period
num_steps = np.int(np.ceil(T / dt))

# Create model
eqn = LinearWaveEquation(mesh, k=degree, omega=omega, c=c, c0=c0, lumped=True)

# Plot the field on the central axis
# n_line = np.int(np.ceil(wx/lam * 20))
# x_line = np.linspace(-wx/2, wx/2, n_line)
# points = np.vstack((x_line, np.zeros(n_line), np.zeros(n_line)))

Nx = np.int(np.ceil(wx/lam * 10))
Ny = Nx
xmin, xmax, ymin, ymax = [-wx/2, wx/2, -wy/2, wy/2]
plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
points = np.vstack((plot_grid[0].ravel(),
                    plot_grid[1].ravel(),
                    np.zeros(plot_grid[0].size)))
points_2d = points[0:2, :]

# Solve
u, u_quad = RK_eval_split_field.solve(eqn.f0, eqn.f1, *eqn.init(), dt=dt,
                              num_steps=num_steps, points=points,
                              n_period=n_period, mesh=mesh, incident=incident)

# Compute error using known analytical solution

# Time-averaged absolute pressure
p_squared_av = 1/period * dt * u_quad
p_squared = 2 * p_squared_av
out_domain = points_2d[0]**2 + points_2d[1]**2 >= (wx/2)**2
p_squared[out_domain] = 0.0
p_sq = p_squared.reshape((Nx, Ny))

# Plotting
if (MPI.COMM_WORLD.rank == 0):
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 22})
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    plt.imshow(np.fliplr(p_sq).T,
               extent=[-wx/2, wx/2, -wy/2, wy/2],
               cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Draw circular scatterer
    circle = plt.Circle((0., 0.), rad_circ, color='black', fill=False)
    ax.add_artist(circle)

    plt.colorbar()
    fig.savefig('results/p_squared.png')
    plt.close()

# Compare against analytical solution
k0 = omega / c0
k1 = refInd * k0
u_exact = penetrable_circle(k0, k1, rad_circ, plot_grid)
u_exact[out_domain.reshape((Nx, Ny))] = 0.0

if (MPI.COMM_WORLD.rank == 0):
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 22})
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    plt.imshow(np.fliplr(np.abs(u_exact)**2-p_sq).T,
               extent=[-wx/2, wx/2, -wy/2, wy/2],
               cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Draw circular scatterer
    circle = plt.Circle((0., 0.), rad_circ, color='black', fill=False)
    ax.add_artist(circle)

    plt.colorbar()
    fig.savefig('results/p_difference.png')
    plt.close()

relative_error = np.linalg.norm(np.abs(u_exact)**2 - p_sq) / \
                np.linalg.norm(np.abs(u_exact)**2)
if (MPI.COMM_WORLD.rank == 0):
    print('Relative error = ', relative_error)
