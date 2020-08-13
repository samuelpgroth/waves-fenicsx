#
# Time harmonic scattering by a penetrable circle or sphere
# =========================================================
#
# We consider a plane wave scattered by a penetrable circle or sphere (with
# the option to switch between the two). The outgoing scattered field is
# absorbed by a first order absorbing boundary condition at the exterior
# boundary - this yields some reflections but is sufficiently good for <5%
# error. The finite element approximation is compared to the analytical
# solution at the end to check the accuracy.

import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI
import ufl
import time
import dolfinx
from ufl import (FiniteElement, inner, grad, TestFunction, TrialFunction, ds,
                 dx, FacetNormal)
from dolfinx import FunctionSpace, Function, geometry
from petsc4py import PETSc
from evaluate_fields import evaluate

'''                         Problem parameters                             '''
# Prescribe:
# * frequency (f0) of the incident plane wave
# * direction (direc) of the incident plane wave
# * radius of the circle/sphere
# * refraction index of scatterer (ref_index)
# * choose 2D or 3D (i.e., circle or sphere)
# (NB: direction is irrelevant for the circle/sphere case, but it's helpful if
# the user wants to change to a non-symmetric scatterer).

# Frequency (Hz)
f0 = 1.1e6
# Direction of propagation of incident plane wave
angle = 0
direc = [np.cos(angle), np.sin(angle), 0.0]
direc = direc / np.linalg.norm(direc)
radius = 0.001
# Refractive index
ref_index = 1.1

# Wavespeed (m/s)
c0 = 1.48e3
# Medium density (kg/m^3)
rho = 1.02e3
# Wavenumber
k0 = 2 * np.pi * f0 / c0
# Wavelength (m)
wavelength = 2 * np.pi / k0
# 2D or 3D problem:
dimension = 3

print('Size parameter = ',  float('%.3g' % (k0 * radius)))

'''                       Discretisation parameters                         '''
# Prescribe the following finite element discretisation parameters:
# * number of mesh elements per wavelength (n_per_wave)
# * polynomial degree (degree)
n_per_wave = 3
degree = 2

h = wavelength / n_per_wave
domain_radius = radius + 3 * wavelength


def circle_mesh():
    import pygmsh, meshio
    if MPI.COMM_WORLD.rank == 0:
        geom = pygmsh.built_in.Geometry()
        disk_inner = geom.add_circle([0, 0, 0], radius, h/ref_index)
        disk = geom.add_circle([0, 0, 0], domain_radius, h,
                               holes=[disk_inner.line_loop])
        mesh = pygmsh.generate_mesh(geom, dim=2, prune_z_0=True)
        cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                    if cells.type == "triangle"]))
        triangle_mesh = meshio.Mesh(points=mesh.points,
                                    cells=[("triangle", cells)])

        # Write mesh
        meshio.xdmf.write("mesh.xdmf", triangle_mesh)


def ellipsoid_mesh(a, b, c, h):
    import pygmsh, meshio
    import numpy as np
    if MPI.COMM_WORLD.rank == 0:
        geom = pygmsh.built_in.Geometry()

        geom.add_ellipsoid([0, 0, 0], [a, b, c], h)

        mesh = pygmsh.generate_mesh(geom)
        cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                    if cells.type == "tetra"]))
        triangle_mesh = meshio.Mesh(points=mesh.points,
                                    cells=[("tetra", cells)])

        # Write mesh
        meshio.xdmf.write("mesh.xdmf", triangle_mesh)


if dimension==2:
    circle_mesh()
elif dimension==3:
    ellipsoid_mesh(domain_radius, domain_radius, domain_radius, h)

# Read mesh
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r",
              XDMFFile.Encoding.HDF5) as xdmf:
         mesh = xdmf.read_mesh(name="Grid")

n = FacetNormal(mesh)

x = ufl.geometry.SpatialCoordinate(mesh)
if dimension==2:
    di = ufl.as_vector([np.cos(angle), np.sin(angle)])
elif dimension==3:
    di = ufl.as_vector([np.cos(angle), np.sin(angle), 0])
ui = ufl.exp(1j * k0 * ufl.dot(di, x))

# Create function space
V = FunctionSpace(mesh, FiniteElement("Lagrange", mesh.ufl_cell(), degree))

def circle_refractive_index(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    inside  = ( r <= radius )
    outside = ( r > radius )
    return inside * ref_index * k0 + outside * k0

def sphere_refractive_index(x):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    inside  = ( r <= radius )
    outside = ( r > radius )
    return inside * ref_index * k0 + outside * k0

V_DG = FunctionSpace(mesh, ("DG", 0))
k = Function(V_DG)
if dimension==2:
    k.interpolate(circle_refractive_index)
elif dimension==3:
    k.interpolate(sphere_refractive_index)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

duidn = ufl.dot(grad(ui), n)

# Scattered field formulation
a = inner(grad(u), grad(v)) * dx \
    - k**2 * inner(u, v) * dx \
    - 1j * k * inner(u, v) * ds \
    # + 0.5*inner(u/distance, v) * ds  # 2nd-order ABC term


L = -inner(grad(ui), grad(v)) * dx \
    + k**2 * inner(ui, v) * dx \
    + inner(duidn, v) * ds

A = dolfinx.fem.assemble_matrix(a)
A.assemble()

solver = PETSc.KSP().create(mesh.mpi_comm())
opts = PETSc.Options()

# Direct solve with MUMPS
# opts["ksp_type"] = "preonly"
# opts["pc_type"] = "lu"
# opts["pc_factor_mat_solver_type"] = "mumps"
opts["ksp_type"] = "gmres"
opts["ksp_rtol"] = 1.0e-5
opts["ksp_view"] = ""
opts["pc_type"] = "asm"


solver.setFromOptions()
solver.setOperators(A)

b = dolfinx.fem.assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, 
              mode=PETSc.ScatterMode.REVERSE)

# Compute solution
u = Function(V)
start = time.time()
solver.solve(b, u.vector)
end = time.time()
time_elapsed = end - start
print('Solve time: ', time_elapsed)
print('Iterations = ', solver.its)
u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                             mode=PETSc.ScatterMode.FORWARD)

with XDMFFile(MPI.COMM_WORLD, "sol.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u)

# Evaluate finite element approximation and compare to analytical solution
Nx = np.int(np.ceil(2*domain_radius/wavelength * 10))
Ny = Nx
xmin, xmax, ymin, ymax = [-domain_radius, domain_radius,
                          -domain_radius, domain_radius]
plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
points = np.vstack((plot_grid[0].ravel(),
                    plot_grid[1].ravel(),
                    np.zeros(plot_grid[0].size)))
u_eval = evaluate(points, mesh, u).reshape((Nx, Ny))


def incident(x):
    return np.exp(1.0j * k0 * (np.cos(angle) * x[0] + np.sin(angle) * x[1]))


out_domain = points[0]**2 + points[1]**2 >= (domain_radius)**2

inc_field = incident(points)
u_inc = inc_field.reshape((Nx, Ny))
u_inc[out_domain.reshape((Nx, Ny))] = 0.0
# Plotting
import matplotlib
if (MPI.COMM_WORLD.rank == 0):
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 22})
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    plt.imshow(np.fliplr(np.real(u_eval+u_inc)).T,
               extent=[-domain_radius, domain_radius, -domain_radius,
                       domain_radius],
               cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Draw circular scatterer
    circle = plt.Circle((0., 0.), radius, color='black', fill=False)
    ax.add_artist(circle)

    plt.colorbar()
    fig.savefig('helmholtz.png')
    plt.close()


# Compare against analytical solution
if dimension==2:
    k1 = ref_index * k0
    from analytical import penetrable_circle
    u_exact = penetrable_circle(k0, k1, radius, plot_grid)
elif dimension==3:
    from analytical import sphere_density_contrast
    u_exact = sphere_density_contrast(k0 * radius, ref_index, Nx, rho, rho, plot_grid, radius)

u_exact[out_domain.reshape((Nx, Ny))] = 0.0

error = np.linalg.norm(np.conj(u_exact)-u_eval-u_inc)/np.linalg.norm(u_exact) 
print('Relative error = ', error)


if (MPI.COMM_WORLD.rank == 0):
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 22})
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    plt.imshow(np.fliplr(np.real(u_exact)).T,
               extent=[-domain_radius, domain_radius, -domain_radius,
                       domain_radius],
               cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Draw circular scatterer
    circle = plt.Circle((0., 0.), radius, color='black', fill=False)
    ax.add_artist(circle)

    plt.colorbar()
    fig.savefig('helmholtz3d.png')
    plt.close()


# from IPython import embed; embed()

