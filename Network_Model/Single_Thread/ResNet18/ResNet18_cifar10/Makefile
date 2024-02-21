main : main.o resnet18.o
	g++ -o main main.o resnet18.o -lm -fopenmp

main.o : main.cpp
	g++ -c main.cpp -fopenmp

resnet18.o : resnet18.cpp resnet18.h
	g++ -c resnet18.cpp -fopenmp

clean :
	rm main main.o resnet18.o