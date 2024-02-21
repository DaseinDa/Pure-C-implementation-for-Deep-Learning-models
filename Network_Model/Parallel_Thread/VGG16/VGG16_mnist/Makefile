main : main.o vgg.o
	g++ -o main main.o vgg.o -lm -fopenmp

main.o : main.cpp
	g++ -c main.cpp -fopenmp

vgg.o : vgg.cpp vgg.h
	g++ -c vgg.cpp -fopenmp

clean :
	rm main main.o vgg.o