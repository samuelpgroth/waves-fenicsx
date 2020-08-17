import numpy as np
import time
import matplotlib
import dolfinx
from mpi4py import MPI
from dolfinx import (Function, FunctionSpace, RectangleMesh,
                     geometry, NewtonSolver)
from dolfinx.io import XDMFFile
from dolfinx.fem import assemble_vector
from dolfinx.cpp.mesh import CellType
from ufl import (ds, dx, grad, inner, dot, derivative,
                 FiniteElement, TestFunction, TrialFunction, lhs, rhs)
from petsc4py import PETSc

# import dolfinx.log
# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# Set parameters for simulation
degree = 4              # degree of polynomials in approximation space
time_step_method = 'CN'  # CN = Crank-Nicolson, RK = Runge-Kutta

# Incident wavefield parameters
omega = 2*np.pi*1e6           # angular frequency
delta0 = 16.0e2          # attenuation
c0 = 1.48e3              # wavespeed
p0 = 1.0             # initial pressure amplitude

# Generate useful parameters
f0 = omega / (2*np.pi)  # frequency
lam = c0 / f0           # wavelength
period = 1.0 / f0       # temporal period
k0 = omega / c0         # wavenumber

t, T = 0, 16*period           # simulation start and end times
CFL = 0.25               # CFL constant


# dim_x = 14*lam            # width of computational domain
n_per_lam = 4
rad_circ = 2*lam
rad_dom = rad_circ + 4 * lam
refInd = 1.2
dim_x = rad_dom * 2  # radius diameter

h_elem = lam / n_per_lam
n_elem_x = np.int(np.round(dim_x/h_elem))

dpml = 0*lam

# Direction of propagation of incident beam
inc_ang = 0
d_inc = [np.cos(inc_ang), np.sin(inc_ang)]
d_inc = d_inc / np.linalg.norm(d_inc)

# Generate mesh
def circle_mesh():
    import pygmsh, meshio
    geom = pygmsh.built_in.Geometry()
    disk_inner = geom.add_circle([0, 0, 0], rad_circ, h_elem/refInd)
    disk = geom.add_circle([0, 0, 0], dim_x/2, h_elem,
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