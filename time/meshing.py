# Repository for pygmsh meshing functions

from dolfinx.cpp.io import perm_gmsh
from mpi4py import MPI
import pygmsh
import numpy as np
from dolfinx.io import ufl_mesh_from_gmsh
import dolfinx


def build_cylinder(n):
    """Build hex cylinder mesh using gmsh"""
    if MPI.COMM_WORLD.rank == 0:
        h = 1.0 / n
        geom = pygmsh.opencascade.Geometry()
        geom.add_raw_code("Mesh.RecombineAll = 1;")
        geom.add_raw_code("Mesh.CharacteristicLengthMin = {};".format(0.98 * h))
        geom.add_raw_code("Mesh.CharacteristicLengthMax = {};".format(1.02 * h))
        geom.add_raw_code("Mesh.Algorithm = 2;")
        geom.add_raw_code("Mesh.RecombinationAlgorithm = 1;")
        circle = geom.add_disk([0.0, 0.0, 0.0], 0.5)
        geom.rotate(circle, [0, 0, 0], np.pi / 2, [0, 1, 0])
        geom.extrude(circle, translation_axis=[1, 0.0, 0.0], num_layers=n, recombine=True)
        pygmsh_mesh = pygmsh.generate_mesh(geom)
        cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
        num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    else:
        num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
        cells, x = np.empty((0, num_nodes)), np.empty((0, 3))

    domain = ufl_mesh_from_gmsh("hexahedron", 3)
    gmsh_hex8 = perm_gmsh(dolfinx.cpp.mesh.CellType.hexahedron, 8)
    return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells[:, gmsh_hex8], x, domain)


def build_piston(n, l, r):
    """Build hex cylinder mesh using gmsh"""
    if MPI.COMM_WORLD.rank == 0:
        h = l / n
        geom = pygmsh.opencascade.Geometry()
        geom.add_raw_code("Mesh.RecombineAll = 1;")
        # geom.add_raw_code("Mesh.RecombineAll = {};".format(l))
        geom.add_raw_code("Mesh.CharacteristicLengthMin = {};".format(0.98 * h))
        geom.add_raw_code("Mesh.CharacteristicLengthMax = {};".format(1.02 * h))
        geom.add_raw_code("Mesh.Algorithm = 2;")
        geom.add_raw_code("Mesh.RecombinationAlgorithm = 1;")
        circle = geom.add_disk([0.0, 0.0, 0.0], r)
        geom.rotate(circle, [0, 0, 0], np.pi / 2, [0, 1, 0])
        geom.extrude(circle, translation_axis=[l, 0.0, 0.0], num_layers=n, recombine=True)
        pygmsh_mesh = pygmsh.generate_mesh(geom)
        cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
        num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    else:
        num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
        cells, x = np.empty((0, num_nodes)), np.empty((0, 3))
    # from IPython import embed; embed()
    domain = ufl_mesh_from_gmsh("hexahedron", 3)
    gmsh_hex8 = perm_gmsh(dolfinx.cpp.mesh.CellType.hexahedron, 8)
    return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells[:, gmsh_hex8], x, domain)
