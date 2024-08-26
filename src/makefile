NVCC = nvcc
CUDAFLAGS = -Icuda

%-cu.o: cuda/%.cu
	$(NVCC) -c -o $@ $< $(CUDAFLAGS)

hvm-cu: hvm-cu.o
	$(NVCC) $(CUDAFLAGS) -o hvm-cu $^
