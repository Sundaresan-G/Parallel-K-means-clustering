all: main.out

main.out: device.o host.o
	g++ -o main.out host.o device.o -lcudart
	
device.o: device.cu
	nvcc -o device.o device.cu

host.o: host.cpp
	g++ -o host.o host.cpp 

clean:
	rm -f main.out host.o device.o
