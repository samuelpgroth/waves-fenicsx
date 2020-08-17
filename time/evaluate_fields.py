from dolfinx import geometry
import numpy as np
from mpi4py import MPI


def evaluate(points, mesh, u):
    tree = geometry.BoundingBoxTree(mesh, mesh.geometry.dim)

    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    colliding_cells = -np.ones(points.shape[1], dtype=np.int32)
    for i, point in enumerate(points.T):
        # Find first colliding cell
        colliding_cell = geometry.compute_colliding_cells(tree, mesh, point, 1)
        # Only add cell to list if it is owned by the processor
        if len(colliding_cell) > 0 and colliding_cell[0] < num_local_cells:
            colliding_cells[i] = colliding_cell[0]

    local_cells = np.argwhere(colliding_cells != -1).T[0]
    on_proc = np.zeros(colliding_cells.shape[0])
    on_proc[local_cells] = 1
    # Workaround since the cell exists on multiple processors, not respecting
    # ghosting.
    num_proc = MPI.COMM_WORLD.allgather(on_proc)
    # from IPython import embed; embed()

    u_on_proc = u.eval(points.T, colliding_cells)
    u_g = MPI.COMM_WORLD.allgather(u_on_proc)
    u_gathered = sum(u_g).T[0] #/ sum(num_proc)

    return u_gathered
