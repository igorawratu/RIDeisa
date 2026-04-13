from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print("R = " + str(rank))