import numpy as np
from gpaw.mpi import serial_comm


def parallel_partition(N, commrank, commsize):

    res =  N % commsize

    if (commrank < res):
        N_local =  N // commsize + 1
        N_start = commrank * N_local
        N_end = (commrank + 1) * N_local
    else:
        N_local = N // commsize
        N_start = commrank * N_local + res
        N_end = (commrank + 1) * N_local + res

    if commrank == commsize - 1:
        N_end = N

    return N, N_local, N_start, N_end
