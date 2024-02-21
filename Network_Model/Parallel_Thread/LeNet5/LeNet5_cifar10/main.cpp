#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdio>
#include "data/cifar/cifar10_reader.hpp"

#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		30000
#define COUNT_TEST		1
#define ITERATION		1

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))
decltype(std::chrono::system_clock::now()) start;
decltype(std::chrono::system_clock::now() - std::chrono::system_clock::now()) duration;

void start_timer() {
	start = std::chrono::system_clock::now();
}
void stop_timer() {
	duration = std::chrono::system_clock::now() - start;
}


void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		TrainBatch(lenet, train_data + i, train_label + i, batch_size);
		// Feature features = { 0 };
		// int right = 0;
		// uint8 l = train_label[i];
		// int p = Predict(lenet, train_data[i], 10);
		// right += l == p;
		// if (i * 100 / total_size > percent)
		// 	printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = right * 100 / total_size);
	}
}

// int testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
// {
// 	int right = 0, percent = 0;
// 	for (int i = 0; i < total_size; ++i)
// 	{
// 		uint8 l = test_label[i];
// 		int p = Predict(lenet, test_data[i], 10);
// 		right += l == p;
// 		if (i * 100 / total_size > percent)
// 			printf("test:%2d%%\n", percent = right * 100 / total_size);
// 	}
// 	return right;
// }

int save(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

int load(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}



void foo()
{

	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	// image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	// uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));

	// LeNet5 lenet_test={0};
	// printf("The number of LeNet5 Mnist double network parameters is:%d\n",GETCOUNT(lenet_test));
	// sleep(30);
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	std::cout <<"Data loader work success"<< std::endl;
    /*读取数据*/
    memcpy(train_data,&dataset.training_images[0],COUNT_TRAIN*sizeof(image));
    memcpy(train_label,&dataset.training_labels[0],COUNT_TRAIN*sizeof(uint8));

	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	Initial(lenet);
	int batches[] = { 1000 };
	start_timer();
	for(int epoch =0;epoch < ITERATION;epoch++){
		for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
			training(lenet, train_data, train_label, batches[i],COUNT_TRAIN);
		// int right = testing(lenet, test_data, test_label, COUNT_TEST);
		// printf("%d/%d\n", right, COUNT_TEST);
		// printf("Time:%u\n", (unsigned)(clock() - start));
}
	stop_timer();
	std::cout << "The time of an epoch when batch=300 image trainng of LeNet5 cifar single is " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << std::endl;
	//save(lenet, LENET_FILE);
	free(lenet);
	free(train_data);
	free(train_label);
	system("pause");
}

int main()
{
// #pragma omp parallel num_threads(4)
//   {
//     printf("hello world from tid = %d\n", omp_get_thread_num()); // omp_get_thread_num 表示得到线程的线程 id
//   }


// omp_set_num_threads(20); //设置线程的个数
// double start = omp_get_wtime();//获取起始时间  
// #pragma omp parallel for
// for (int i = 0; i < 50; i++) {
// 	for (int j = 0; j < 10; j++) {
// 		printf("%d %d\n",i,j);
// 	}

// }
	foo();
	return 0;
}