from mpi4py import MPI
import sys
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

bla,blub = sys.argv[1:]

print bla,blub
if rank == 0:
    data = {'a':7,'b' : 3.14}
    comm.send(data,dest=1,tag=11)
    print("sent!")
elif rank == 1:
    data = comm.recv(source=0,tag=11)
    print('received:\n{}'.format(data))


