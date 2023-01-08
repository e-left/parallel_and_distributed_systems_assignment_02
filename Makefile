CPPFLAGS = -I /opt/homebrew/opt/openblas/include -lblas

sequential: $(wildcard sequential/*.c)
	gcc $(CPPFLAGS) $(wildcard sequential/*.c) -o out/sequential

mpi: $(wildcard asynchronous_mpi/*.c)
	mpicc $(CPPFLAGS) $(wildcard asynchronous_mpi/*.c) -o out/mpi