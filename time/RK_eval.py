import time
import numpy as np
from mpi4py import MPI
from dolfinx.io import XDMFFile
from petsc4py import PETSc
from evaluate_fields import evaluate


def solve(f0, f1, u, v, dt, num_steps, points, n_period, mesh):
    """Solve problem using the Runge Kutta method"""

    # Create solution vectors at RK intermediate stages
    un, vn = u.vector.copy(), v.vector.copy()

    # Solution at start of time step
    u0, v0 = u.vector.copy(), v.vector.copy()

    # Get Runge-Kutta timestepping data
    n_RK, a_runge, b_runge, c_runge = butcher(4)

    # Create lists to hold intermediate vectors
    ku, kv = n_RK * [None], n_RK * [None]

    # file = XDMFFile(MPI.COMM_WORLD, "u.xdmf", "w")
    # file.write_mesh(u.function_space.mesh)
    # file.write_function(u, t=0.0)

    u_on_axis = np.zeros([points.T.shape[0], n_period])
    counter = -1

    # print("Num dofs:", u.vector.size)
    t = 0.0
    # loop_start = time.time()
    for step in range(num_steps):
        # print("Time step:", step, t, dt)

        # Store solution at start of time step as u0 and v0
        u0 = u.vector.copy(result=u0)
        v0 = v.vector.copy(result=v0)

        # Runge Kutta steps,
        # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods
        for i in range(n_RK):
            un = u0.copy(result=un)
            vn = v0.copy(result=vn)
            for j in range(i):
                a = dt * a_runge[i, j]
                un.axpy(a, ku[j])
                vn.axpy(a, kv[j])

            # RK evaluation time
            tn = t + c_runge[i] * dt

            # Compute RHS vector
            ku[i] = f0(tn, un, vn, result=ku[i])
            kv[i] = f1(tn, un, vn, result=kv[i])

            # Update solution
            u.vector.axpy(dt * b_runge[i], ku[i])
            v.vector.axpy(dt * b_runge[i], kv[i])

        # Update time
        t += dt
        # if step % 10 == 0:
        #     file.write_function(u, t=t)

        if step >= num_steps - n_period:
            counter += 1
            u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                 mode=PETSc.ScatterMode.FORWARD)
            u_on_axis[:, counter] = evaluate(points, mesh, u)

    # end = time.time()
    # print("RK time:", end - loop_start)
    # print("RK time per step:", (end - loop_start) / num_steps)

    return u, u_on_axis


def butcher(order):
    """Butcher table data"""
    if order == 2:
        # Explicit trapezium method
        n_RK = 2
        b_runge = [1 / 2, 1 / 2]
        c_runge = [0, 1]
        a_runge = np.zeros((n_RK, n_RK), dtype=np.float)
        a_runge[1, 0] = 1
    elif order == 3:
        # Third-order RK3
        n_RK = 3
        b_runge = [1 / 6, 4 / 6, 1 / 6]
        c_runge = [0, 1 / 2, 1]
        a_runge = np.zeros((n_RK, n_RK), dtype=np.float)
        a_runge[1, 0] = 1 / 2
        a_runge[2, 0] = -1
        a_runge[2, 1] = 2
    elif order == 4:
        # "Classical" 4th-order Runge-Kutta method
        n_RK = 4
        b_runge = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
        c_runge = np.array([0, 1 / 2, 1 / 2, 1])
        a_runge = np.zeros((4, 4), dtype=np.float)
        a_runge[1, 0] = 1 / 2
        a_runge[2, 1] = 1 / 2
        a_runge[3, 2] = 1
    elif order == 5:
        # Fehlberg 5th order
        n_RK = 6
        b_runge = [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]
        c_runge = [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]
        a_runge = np.zeros((n_RK, n_RK), dtype=np.float)
        a_runge[1, 0] = 1 / 4
        a_runge[2, 0:2] = [3 / 32, 9 / 32]
        a_runge[3, 0:3] = [1932 / 2197, -7200 / 2197, 7296 / 2197]
        a_runge[4, 0:4] = [439 / 216, -8, 3680 / 513, -845 / 4104]
        a_runge[5, 0:5] = [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]

    return n_RK, a_runge, b_runge, c_runge
