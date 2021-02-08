#
# Scattering of a plane wave by a sound-hard circle using high-order mesh
# =======================================================================
#
# This demo illustrates how to:
#
# * Compute the scattering of a plane wave by a sound-hard circle
# * Use quadratic mesh elements to represent the circular boundary accurately
# * Employ an "adiabatic layer" to truncate the domain
# * Evaluate the FEM solution at specified grid points
# * Compare the approximation to the analytical solution
# * Make a nice plot of the solution in the domain
#
# Adiabatic absorbers are presented in detail in
# "The failure of perfectly matched layers, and towards their redemption
#  by adiabatic absorbers" - Oskooi et al. (2008)

import numpy as np
from mpi4py import MPI
import gmsh
import matplotlib.pyplot as plt
import matplotlib
from gmsh_helpers import gmsh_model_to_mesh
import time
from petsc4py import PETSc
from dolfinx import (Function, FunctionSpace, geometry, has_petsc_complex)
from ufl import (dx, grad, inner, dot, TestFunction, TrialFunction,
                 FacetNormal, Measure, lhs, rhs)
from dolfinx.io import XDMFFile
from analytical import sound_hard_circle
from dolfinx.mesh import locate_entities_boundary
import dolfinx

# This implementation relies on the complex mode of dolfin-x, invoked by
# executing the command:
# source /usr/local/bin/dolfinx-complex-mode
if not has_petsc_complex:
    print('This demo only works with PETSc-complex')
    exit()


'''                        Problem parameters                               '''
k0 = 10                  # wavenumber
wave_len = 2*np.pi / k0  # wavelength
radius = 1 * wave_len    # scatterer radius
d_air = 4 * wave_len     # distance between scatterer and absorbing layer

'''    Discretization parameters: polynomial degree and mesh resolution     '''
degree = 3  # polynomial degree
n_wave = 5  # number of mesh elements per wavelength

'''                   Adiabatic absorber settings                           '''
# The adiabatic absorber is a PML-type layer in which absorption is used to
# attenutate outgoing waves. Adiabatic absorbers aren't as perfect as PMLs so
# must be slightly wider: typically 2-5 wavelengths gives adequately small
# reflections.
d_absorb = 2 * wave_len    # depth of absorber

# Increase the absorption within the layer gradually, as a monomial:
# sigma(x) = sigma_0 * x^d; choices d=2,3 are popular choices.
deg_absorb = 2    # degree of absorption monomial

# The constant sigma_0 is chosen to achieve a specified "round-trip" reflection
# of a wave that through the layer, reflects and returns back into the domain.
# See Oskooi et al. (2008) for more details.
RT = 1.0e-6       # round-trip reflection
sigma0 = -(deg_absorb + 1) * np.log(RT) / (2.0 * d_absorb)

'''                             Meshing                                     '''
# For this problem we use a square mesh with triangular elements.
# The domain width is 2 * (radius + d_air + d_absorb)
dim_x = 2 * (radius + d_air + d_absorb)

# The mesh element size is h_elem
h_elem = wave_len / n_wave

# We create the mesh using the Gmsh Python API. For a tutorial, see
# http://jsdokken.com/converted_files/tutorial_gmsh.html
gmsh.initialize()
gdim = 2  # dimension of the problem

# The mesh is a square with a disk cut out
rank = MPI.COMM_WORLD.rank
if rank == 0:
    rectangle = gmsh.model.occ.addRectangle(-dim_x/2, -dim_x/2, 0,
                                            dim_x, dim_x, tag=1)
    obstacle = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()

fluid_marker = 1
if rank == 0:
    volumes = gmsh.model.getEntities(dim=gdim)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h_elem)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.model.mesh.setOrder(2)  # Command required for quadratic elements

# Use a gmsh helper function to create the mesh. This requires Jorgen's
# file http://jsdokken.com/converted_files/gmsh_helpers.py
mesh, cell_tags = gmsh_model_to_mesh(gmsh.model, cell_data=True, gdim=2)
n = FacetNormal(mesh)


def boundary(x):
    return (x[0]**2 + x[1]**2) < (radius+1.0e-2)**2


circle_facets = locate_entities_boundary(mesh, 1, boundary)
mt = dolfinx.mesh.MeshTags(mesh, 1, circle_facets, 1)

ds = Measure("ds", subdomain_data=mt)

'''        Incident field, wavenumber and adiabatic absorber functions      '''


def incident(x):
    # Plane wave travelling in positive x-direction
    return np.exp(1.0j * k0 * x[0])


def adiabatic_layer(x):
    '''          Contribution to wavenumber k in absorbing layers          '''
    # In absorbing layer, have k = k0 + 1j * sigma
    # => k^2 = (k0 + 1j*sigma)^2 = k0^2 + 2j*sigma - sigma^2
    # Therefore, the 2j*sigma - sigma^2 piece must be included in the layer.

    # Find borders of width d_absorb in x- and y-directions
    in_absorber_x = (np.abs(x[0]) >= dim_x/2 - d_absorb)
    in_absorber_y = (np.abs(x[1]) >= dim_x/2 - d_absorb)

    # Function sigma_0 * x^d, where x is depth into adiabatic layer
    sigma_x = sigma0 * ((np.abs(x[0])-(dim_x/2-d_absorb))/d_absorb)**deg_absorb
    sigma_y = sigma0 * ((np.abs(x[1])-(dim_x/2-d_absorb))/d_absorb)**deg_absorb

    # 2j*sigma - sigma^2 in absorbing layers
    x_layers = in_absorber_x * (2j * sigma_x * k0 - sigma_x**2)
    y_layers = in_absorber_y * (2j * sigma_y * k0 - sigma_y**2)

    return x_layers + y_layers


# Define function space
V = FunctionSpace(mesh, ("Lagrange", degree))

# Interpolate absorbing layer piece of wavenumber k_absorb onto V
k_absorb = Function(V)
k_absorb.interpolate(adiabatic_layer)

# Interpolate incident wave field onto V
ui = Function(V)
ui.interpolate(incident)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

F = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx - \
    k_absorb * inner(u, v) * dx \
    + inner(dot(grad(ui), n), v) * ds(1)

a = lhs(F)
L = rhs(F)

'''           Assemble matrix and vector and set up direct solver           '''
A = dolfinx.fem.assemble_matrix(a)
A.assemble()
b = dolfinx.fem.assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

solver = PETSc.KSP().create(mesh.mpi_comm())
opts = PETSc.Options()
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
solver.setFromOptions()
solver.setOperators(A)

# Solve linear system
u = Function(V)
start = time.time()
solver.solve(b, u.vector)
end = time.time()
time_elapsed = end - start
print('Solve time: ', time_elapsed)
u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                     mode=PETSc.ScatterMode.FORWARD)

# Write solution to file
with XDMFFile(MPI.COMM_WORLD, "results/sol.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u)

'''            Evaluate field over a specified grid of points              '''
# Square grid with 10 points per wavelength in each direction
Nx = np.int(np.ceil(dim_x/wave_len * 10))
Ny = Nx

# Grid does not include absorbing layers
dim_in = dim_x - 2 * d_absorb

# Grid points
xmin, xmax, ymin, ymax = [-dim_in/2, dim_in/2, -dim_in/2, dim_in/2]
plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
points = np.vstack((plot_grid[0].ravel(),
                    plot_grid[1].ravel(),
                    np.zeros(plot_grid[0].size)))

points_2d = points[:2, :]

# Locate grid points inside the circle. These are outside the computation
# domain so we will ultimately set the field here to zero.
in_circ = points[0, :]**2 + points[1, :]**2 <= (radius)**2
in_circ_2d = points_2d[0, :]**2 + points_2d[1, :]**2 <= (radius)**2
points[0, in_circ] = -radius - wave_len / 10
points[1, in_circ] = radius + wave_len / 10
points[2, in_circ] = 0.

# Bounding box tree etc for function evaluations
tree = geometry.BoundingBoxTree(mesh, 2)
cell_candidates = [geometry.compute_collisions_point(tree, xi)
                   for xi in points.T]
cells = [dolfinx.cpp.geometry.select_colliding_cells(mesh, cell_candidates[i],
         points.T[i], 1)[0] for i in range(len(cell_candidates))]

# Evaluate scattered and incident fields at grid points
u_sca_temp = u.eval(points.T, cells)
u_sca_temp[in_circ_2d] = 0.0            # Set field inside circle to zero
u_sca = u_sca_temp.reshape((Nx, Ny))    # Reshape
inc_field = incident(points_2d)
inc_field[in_circ_2d] = 0.0             # Set field inside circle to zero
u_inc = inc_field.reshape((Nx, Ny))

# Sum to give total field
u_total = u_inc + u_sca

'''                  Compare against analytical solution                    '''
# Uncomment to perform comparison, takes a few seconds to run
u_exact = sound_hard_circle(k0, radius, plot_grid)
diff = u_exact-u_total
error = np.linalg.norm(diff)/np.linalg.norm(u_exact)
print('Relative error = ', error)

'''                     Plot field and save figure                          '''
matplotlib.rcParams.update({'font.size': 22})
plt.rc('font', family='serif')
fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
plt.imshow(np.fliplr(np.real(u_total)).T,
           extent=[-dim_in/2, dim_in/2, -dim_in/2, dim_in/2],
           cmap=plt.cm.get_cmap('seismic'), interpolation='spline16')
circle = plt.Circle((0., 0.), radius, color='black', fill=False)
ax.add_artist(circle)
plt.axis('off')
plt.colorbar()
fig.savefig('results/circle_scatter.png')
