main : main.o alexnet.o
	cc -o main main.o transformer.o -lm

main.o : main.c
	cc -c main.c

alexnet.o : transformer.c transformer.h
	cc -c transformer.c

clean :
	rm main main.o transformer.o