CUDA_CC=nvcc
CUDA_FLAGS= -g -I /usr/local/cuda/include -I ../../cuda/cub
LINK_CC=g++
LINK_FLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart -ltbb
CC_CPP=g++
CPP_FLAGS= -std=c++11 -Wall -Wpedantic -Werror -Wfatal-errors -g

all: rtree
rtree: rtree.o main.o rtreecuda.o nocuda.o
	$(LINK_CC) main.o rtree.o rtreecuda.o nocuda.o -o rtree -lm $(LINK_FLAGS)
main.o: 
	$(CC_CPP) $(CPP_FLAGS) -c main.cpp -o main.o
rtree.o:
	$(CC_CPP) $(FLAGS) -c rtree.cpp -o rtree.o
rtreecuda.o:
	$(CUDA_CC) $(CUDA_FLAGS) -c rtree.cu -o rtreecuda.o
nocuda.o:
	$(CC_CPP) $(CPP_FLAGS) -c nocuda.cpp -o nocuda.o
clean:
	rm -f *o rtree
