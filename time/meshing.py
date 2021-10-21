import numpy as np
import gmsh

from mpi4py import MPI

from dolfinx import cpp
from dolfinx.cpp.io import perm_gmsh
from dolfinx.io import XDMFFile, extract_gmsh_geometry, ufl_mesh_from_gmsh
from dolfinx.mesh import create_mesh


def build_cylinder(n, write_file=False, fname=""):
    """
    Build a hex cylinder mesh using gmsh
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()

    if MPI.COMM_WORLD.rank == 0:
        model.add("Cylinder")
        model.setCurrent("Cylinder")

        h = 1.0/n

        gmsh.option.setNumber("Mesh.RecombineAll", 2)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.98 * h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.02 * h)

        circle = model.occ.addDisk(0, 0, 0, 0.5, 0.5)
        model.occ.rotate([(2, circle)],
                         0., 0., 0., 0., 1., 0.,
                         np.pi/2)
        model.occ.extrude(
            [(2, circle)],
            1, 0, 0, numElements=[n],
            recombine=True)

        model.occ.synchronize()
        model.mesh.generate(3)

        # sort mesh according to their index in gmsh
        x = extract_gmsh_geometry(model, model.getCurrent())

        # extract cells from gmsh
        element_types, element_tags, node_tags = model.mesh.getElements(dim=3)
        name, dim, order, num_nodes, local_coords, num_first_order_nodes = \
            model.mesh.getElementProperties(element_types[0])

        # broadcast cell type data and geometric dimension
        gmsh_cell_id = MPI.COMM_WORLD.bcast(element_types[0], root=0)

        # get mesh data for dim (0, tdim)
        cells = node_tags[0].reshape(-1, num_nodes)-1

        num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
        gmsh.finalize()
    else:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
        num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
        cells, x = np.empty([0, num_nodes]), np.empty([0, 3])

    # permute the mesh topology from GMSH ordering to DOLFIN-X ordering
    domain = ufl_mesh_from_gmsh(gmsh_cell_id, 3)
    gmsh_hex = perm_gmsh(cpp.mesh.CellType.hexahedron, 8)
    cells = cells[:, gmsh_hex]

    mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
    mesh.name = "cylinder_hex"

    if write_file:
        with XDMFFile(MPI.COMM_WORLD, "{}.xdmf".format(fname), "w") as file:
            file.write_mesh(mesh)

    return mesh


def build_piston(n, length, radius, write_file=False, fname=""):
    """
    Build a hex cylinder mesh using gmsh
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()

    if MPI.COMM_WORLD.rank == 0:
        model.add("Piston")
        model.setCurrent("Piston")

        h = length/n

        gmsh.option.setNumber("Mesh.RecombineAll", 2)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.98 * h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.02 * h)

        circle = model.occ.addDisk(0, 0, 0, radius, radius)
        model.occ.rotate([(2, circle)],
                         0., 0., 0., 0., 1., 0.,
                         np.pi/2)
        model.occ.extrude(
            [(2, circle)],
            length, 0, 0, numElements=[n],
            recombine=True)

        model.occ.synchronize()
        model.mesh.generate(3)

        # sort mesh according to their index in gmsh
        x = extract_gmsh_geometry(model, model.getCurrent())

        # extract cells from gmsh
        element_types, element_tags, node_tags = model.mesh.getElements(dim=3)
        name, dim, order, num_nodes, local_coords, num_first_order_nodes = \
            model.mesh.getElementProperties(element_types[0])

        # broadcast cell type data and geometric dimension
        gmsh_cell_id = MPI.COMM_WORLD.bcast(element_types[0], root=0)

        # get mesh data for dim (0, tdim)
        cells = node_tags[0].reshape(-1, num_nodes)-1

        num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
        gmsh.finalize()
    else:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
        num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
        cells, x = np.empty([0, num_nodes]), np.empty([0, 3])

    # permute the mesh topology from GMSH ordering to DOLFIN-X ordering
    domain = ufl_mesh_from_gmsh(gmsh_cell_id, 3)
    gmsh_hex = perm_gmsh(cpp.mesh.CellType.hexahedron, 8)
    cells = cells[:, gmsh_hex]

    mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
    mesh.name = "piston_hex"

    if write_file:
        with XDMFFile(MPI.COMM_WORLD, "{}.xdmf".format(fname), "w") as file:
            file.write_mesh(mesh)

    return mesh


build_cylinder(10)
build_piston(10, 2, 1)
