﻿#include "resnet18.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdio>
#include "data/cifar/cifar10_reader.hpp"

// #define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
// #define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
// #define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
// #define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		2000
#define COUNT_TEST		100
#define ITERATION		100

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))
decltype(std::chrono::system_clock::now()) start;
decltype(std::chrono::system_clock::now() - std::chrono::system_clock::now()) duration;

void start_timer() {
	start = std::chrono::system_clock::now();
}
void stop_timer() {
	duration = std::chrono::system_clock::now() - start;
}


int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

void training(ResNet18 *resnet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		TrainBatch(resnet, train_data + i, train_label + i, batch_size);
		// int right = 0;
		// uint8 l = train_label[i];
		// int p = Predict(resnet, train_data[i], 10);
		// right += l == p;
		// if (i * 100 / total_size > percent)
		// 	printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = right * 100 / total_size);
	}
}

int testing(ResNet18 *resnet, image *test_data, uint8 *test_label,int total_size)
{
	int right = 0, percent = 0;
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
		int p = Predict(resnet, test_data[i], 10);
		right += l == p;
		if (i * 100 / total_size > percent)
			printf("test:%2d%%\n", percent = right * 100 / total_size);
	}
	return right;
}

int save(ResNet18 *resnet, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(resnet, sizeof(ResNet18), 1, fp);
	fclose(fp);
	return 0;
}

int load(ResNet18 *resnet, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(resnet, sizeof(ResNet18), 1, fp);
	fclose(fp);
	return 0;
}



void foo()
{

	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	// image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	// uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
	// if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	// {
	// 	printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
	// 	free(train_data);
	// 	free(train_label);
	// 	system("pause");
	// }
	// if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	// {
	// 	printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
	// 	free(test_data);
	// 	free(test_label);
	// 	system("pause");
	// }
	// for(int i=0;i<28;i++){
	// 	printf("\n");
	// 	for(int j=0;j<28;j++){
	// 		printf("qqqqq%f",train_data[1][i][j]);
	// 	}
	// }
	// ResNet18 res18_test={0};
	// printf("The number of ResNet18 Mnist double network parameters is:%d\n",GETCOUNT(res18_test));
	// sleep(30);

 	auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	std::cout <<"Data loader work success"<< std::endl;
    /*读取数据*/
    memcpy(train_data,&dataset.training_images[0],COUNT_TRAIN*sizeof(image));
    memcpy(train_label,&dataset.training_labels[0],COUNT_TRAIN*sizeof(uint8));

	ResNet18 *resnet = (ResNet18 *)malloc(sizeof(ResNet18));
	Initial(resnet);
	start_timer();
	int batches[] = { 1000 };
	// for(int epoch =0;epoch < ITERATION;epoch++){
		for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
			training(resnet, train_data, train_label, batches[i],COUNT_TRAIN);

// }
	stop_timer();
	std::cout << "The time of an epoch when batch=1 image trainng of ResNet18 MNist single is " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << std::endl;
	//save(resnet, LENET_FILE);
	// free(resnet);
	// free(train_data);
	// free(train_label);
	// free(test_data);
	// free(test_label);
	// system("pause");
}

int main()
{

	foo();
	return 0;
}