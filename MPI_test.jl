using MPI
MPI.Init()

comm = MPI.COMM_WORLD
nprocs    = MPI.Comm_size(comm)
print("I am rank $(MPI.Comm_rank(comm)) of $(nprocs)\n")
MPI.Barrier(comm)

dims = [2,2]
MPI.Dims_create!(nprocs, dims);
reorder = 1
periodic = [0,0]
comm_cart = MPI.Cart_create(comm, dims, periodic, reorder);

me        = MPI.Comm_rank(comm_cart);
coords    = MPI.Cart_coords(comm_cart);
print("I am rank $(me) of coord $(coords)\n")

NNEIGHBORS_PER_DIM = 2
NDIMS_MPI = 2
neighbors = fill(MPI.PROC_NULL, NNEIGHBORS_PER_DIM, NDIMS_MPI);
disp = 1
for i = 1:NDIMS_MPI
    neighbors[:,i] .= MPI.Cart_shift(comm_cart, i-1, disp);
end
print("I am rank $(me) of $(neighbors)\n")
# mpiexecjl --project -n 4 julia MPI_test.jl
