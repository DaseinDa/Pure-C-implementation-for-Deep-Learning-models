
#pragma once

#define LENGTH_KERNEL	3

#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT			1
#define LAYER0			64
#define LAYER1			64
#define LAYER2			128
#define LAYER3			256
#define LAYER4			512
#define OUTPUT          10

#define Res1 16
#define Res2 8
#define Res3 4
#define Res4 2
#define pool2_size 1

#define ALPHA 0.5
#define PADDING 1

typedef unsigned char uint8;
typedef uint8 image[28][28];

typedef struct ResBlock1
{
	double weight1_1[LAYER0][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight1_2[LAYER1][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];

	double weight2_1[LAYER1][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight2_2[LAYER1][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];

	
	double bias1_1[LAYER1];
	double bias1_2[LAYER1];
	double bias2_1[LAYER1];
	double bias2_2[LAYER1];	
}ResBlock1;
typedef struct ResBlock2
{
	double weight1_1[LAYER1][LAYER2][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight1_2[LAYER2][LAYER2][LENGTH_KERNEL][LENGTH_KERNEL];
	double conv1[LAYER1][LAYER2][1][1];
	double weight2_1[LAYER2][LAYER2][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight2_2[LAYER2][LAYER2][LENGTH_KERNEL][LENGTH_KERNEL];

	double bias1_1[LAYER2];
	double bias1_2[LAYER2];
	double bias2_1[LAYER2];
	double bias2_2[LAYER2];	
}ResBlock2;
typedef struct ResBlock3
{
	double weight1_1[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight1_2[LAYER3][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	double conv1[LAYER2][LAYER3][1][1];
	double weight2_1[LAYER3][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight2_2[LAYER3][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];

	double bias1_1[LAYER3];
	double bias1_2[LAYER3];
	double bias2_1[LAYER3];
	double bias2_2[LAYER3];	
}ResBlock3;
typedef struct ResBlock4
{
	double weight1_1[LAYER3][LAYER4][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight1_2[LAYER4][LAYER4][LENGTH_KERNEL][LENGTH_KERNEL];
	double conv1[LAYER3][LAYER4][1][1];
	double weight2_1[LAYER4][LAYER4][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight2_2[LAYER4][LAYER4][LENGTH_KERNEL][LENGTH_KERNEL];

	double bias1_1[LAYER4];
	double bias1_2[LAYER4];
	double bias2_1[LAYER4];
	double bias2_2[LAYER4];	
}ResBlock4;


typedef struct ResNet18
{
	double weight1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];//Layer1 kernel
	ResBlock1 res_block1;
	ResBlock2 res_block2;
	ResBlock3 res_block3;
	ResBlock4 res_block4;

	double fc[LAYER4*pool2_size*pool2_size][OUTPUT];
	double bias1[LAYER1];
	double bias_fc[OUTPUT];
}ResNet18;


typedef struct Res1_Feature
{
	double conv1_1[LAYER1][Res1][Res1];
	double conv1_2[LAYER1][Res1][Res1];

	double conv2_1[LAYER1][Res1][Res1];
	double conv2_2[LAYER1][Res1][Res1];
}Res1_Feature;


typedef struct Res2_Feature
{
	double conv1_1[LAYER2][Res2][Res2];
	double conv1_2[LAYER2][Res2][Res2];

	double conv_res[LAYER2][Res2][Res2];

	double conv2_1[LAYER2][Res2][Res2];
	double conv2_2[LAYER2][Res2][Res2];
}Res2_Feature;

typedef struct Res3_Feature
{
	double conv1_1[LAYER3][Res3][Res3];
	double conv1_2[LAYER3][Res3][Res3];

	double conv_res[LAYER3][Res3][Res3];

	double conv2_1[LAYER3][Res3][Res3];
	double conv2_2[LAYER3][Res3][Res3];
}Res3_Feature;

typedef struct Res4_Feature
{
	double conv1_1[LAYER4][Res4][Res4];
	double conv1_2[LAYER4][Res4][Res4];

	double conv_res[LAYER4][Res4][Res4];

	double conv2_1[LAYER4][Res4][Res4];
	double conv2_2[LAYER4][Res4][Res4];
}Res4_Feature;





typedef struct Feature
{
	double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
	double conv1[LAYER0][LENGTH_FEATURE0][LENGTH_FEATURE0];
	double pool1[LAYER0][16][16];
	Res1_Feature res1f;
	Res2_Feature res2f;
	Res3_Feature res3f;
	Res4_Feature res4f;

	double pool2[LAYER4][pool2_size][pool2_size];
	double output[OUTPUT];



}Feature;

void TrainBatch(ResNet18 *resnet, image *inputs, uint8 *labels, int batchSize);

void Train(ResNet18 *resnet, image input, uint8 label);

uint8 Predict(ResNet18 *resnet, image input, uint8 count);

void Initial(ResNet18 *resnet);
double relu(double x);