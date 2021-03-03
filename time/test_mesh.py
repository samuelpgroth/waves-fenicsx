import numpy as np
import gmsh

from mpi4py import MPI

from dolfinx import cpp
from dolfinx.cpp.io import perm_gmsh
from dolfinx.io import (XDMFFile, extract_gmsh_geometry,
                        extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh)
from dolfinx.mesh import create_mesh

# incident wavefield parameters
omega = 2 * np.pi * 1e6  # angular frequency
delta0 = 16.0e2          # attenuation
c0 = 1.48e3              # wavespeed
p0 = 1.0                 # initial pressure amplitude

f0 = omega / c0          # frequency
lam = c0 / f0            # wavelength
period = 1.0 / f0        # temporal period
k0 = omega / c0          # wavenumber

# computational domain and mesh parameters
rad_circ = 2 * lam            # scatterer radius
rad_dom = rad_circ + 4 * lam  # domain radius
refInd = 1.2                  # scatterer refractive index
dim_x = 2 * rad_dom           # domain diameter

n_per_lam = 4                 # no. of element per wavelength
h_elem = lam / n_per_lam      # element size
n_elem_x = int(np.round(dim_x/h_elem)) # no. of element on x axis


def circle_mesh(type="quad"):
    """
    Generate a mesh for circular domain with a penetrable circular scatterer.
    Mesh is written in XDMF file format.

    Parameters:
        type : {'quad', 'tri'}
            Type of mesh element

    Returns:
        None
    """

    gmsh.initialize()
    model = gmsh.model()

    model.add("Circular domain")
    model.setCurrent("Circular domain")

    if type == "quad":
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 2)

    elif type == "tri":
        pass

    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", h_elem)

    scatterer = model.occ.addDisk(0, 0, 0, rad_circ, rad_circ)
    domain = model.occ.addDisk(0, 0, 0, dim_x/2, dim_x/2)
    fragment = model.occ.fragment([(2, domain)], [(2, scatterer)])[0]
    model.occ.synchronize()

    model.mesh.generate(2)

    x = extract_gmsh_geometry(model, model.getCurrent())

    element_types, element_tags, node_tags = model.mesh.getElements(dim=2)
    name, dim, order, num_nodes, \
        local_coords, num_first_order_nodes \
        = model.mesh.getElementProperties(element_types[0])
    cells = node_tags[0].reshape(-1, num_nodes)-1

    domain = ufl_mesh_from_gmsh(element_types[0], 2)

    if type == "quad":
        gmsh_quad4 = perm_gmsh(cpp.mesh.CellType.quadrilateral, 4)
        cells = cells[:, gmsh_quad4]
    elif type == "tri":
        pass

    mesh = create_mesh(MPI.COMM_SELF, cells, x[:, :2], domain)
    mesh.name = "Disk"

    with XDMFFile(MPI.COMM_SELF, "disk.xdmf", "w") as file:
        file.write_mesh(mesh)


# generate mesh
circle_mesh("quad")

# read mesh
with XDMFFile(MPI.COMM_WORLD, "disk.xdmf", "r") as file:
    mesh = file.read_mesh(name="Disk")
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(tdim, tdim)
    mesh.topology.create_connectivity(tdim, fdim)
