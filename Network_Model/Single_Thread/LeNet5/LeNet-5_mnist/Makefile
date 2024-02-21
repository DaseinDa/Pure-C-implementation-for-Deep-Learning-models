main : main.o lenet.o
	g++ -o main main.o lenet.o -lm -fopenmp

main.o : main.cpp
	g++ -c main.cpp -fopenmp

lenet.o : lenet.cpp lenet.h
	g++ -c lenet.cpp -fopenmp

clean :
	rm main main.o lenet.o