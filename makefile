CC=clang
FLAGS=-g -std=c99 -Wall -Wpedantic -Werror -Wfatal-errors -g
CUDA_CC=nvcc
CUDA_FLAGS=-g -I /usr/local/cuda/include -I ../../cuda/cub
LINK_CC=clang++
LINK_FLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart

all: rtree
rtree: rtree.o main.o rtreecuda.o
	$(LINK_CC) $(LINK_FLAGS) main.o rtree.o rtreecuda.o -o rtree -lm
main.o: 
	$(CC) $(FLAGS) -c main.c -o main.o
rtree.o:
	$(CC) $(FLAGS) -c rtree.c -o rtree.o
rtreecuda.o:
	$(CUDA_CC) $(CUDA_FLAGS) -c rtree.cu -o rtreecuda.o
clean:
	rm -f *o rtree
