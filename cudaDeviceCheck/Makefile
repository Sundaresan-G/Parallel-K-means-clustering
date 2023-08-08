CC=/apps/cuda/bin/nvcc

all: main

main: cuda_checkdevice.cu
	$(CC) -o main cuda_checkdevice.cu

clean:
	rm -f main
