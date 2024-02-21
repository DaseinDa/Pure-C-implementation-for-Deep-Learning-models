main : main.o alexnet.o
	g++ -o main main.o alexnet.o -lm -fopenmp

main.o : main.cpp
	g++ -c main.cpp -fopenmp

alexnet.o : alexnet.cpp alexnet.h
	g++ -c alexnet.cpp -fopenmp

clean :
	rm main main.o alexnet.o