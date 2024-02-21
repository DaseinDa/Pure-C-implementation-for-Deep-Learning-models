
#pragma once

// #include <charconv>
#define LENGTH_KERNEL	3

#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1 LENGTH_FEATURE0/2  //32
#define LENGTH_FEATURE2 LENGTH_FEATURE1/2  //16
#define LENGTH_FEATURE3 LENGTH_FEATURE2/2	//8
#define LENGTH_FEATURE4 LENGTH_FEATURE3/2	//4
#define LENGTH_FEATURE5 LENGTH_FEATURE4/2	//2



#define INPUT			1
#define LAYER1			64


#define LAYER2			128

#define LAYER3			256


#define LAYER4			512

#define LAYER5			512
#define FC1				4096
#define FC2				4096

#define OUTPUT          10

#define ALPHA 0.5
#define PADDING 1

typedef unsigned char uint8;
typedef uint8 image[28][28];


typedef struct VGG
{
	double weight1_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight1_2[LAYER1][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight2_1[LAYER1][LAYER2][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight2_2[LAYER2][LAYER2][LENGTH_KERNEL][LENGTH_KERNEL];


	double weight3_1[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight3_2[LAYER3][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight3_3[LAYER3][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight3_4[LAYER3][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];


	double weight4_1[LAYER3][LAYER4][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight4_2[LAYER4][LAYER4][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight4_3[LAYER4][LAYER4][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight4_4[LAYER4][LAYER4][LENGTH_KERNEL][LENGTH_KERNEL];

	double weight5_1[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight5_2[LAYER5][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight5_3[LAYER5][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight5_4[LAYER5][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];

	double fc1[LAYER5*LENGTH_FEATURE5*LENGTH_FEATURE5][FC1];
	double fc2[FC1][FC2];
	double fc3[FC2][OUTPUT];

	double bias1_1[LAYER1];
	double bias1_2[LAYER1];

	double bias2_1[LAYER2];
	double bias2_2[LAYER2];

	double bias3_1[LAYER3];
	double bias3_2[LAYER3];
	double bias3_3[LAYER3];
	double bias3_4[LAYER3];

	double bias4_1[LAYER4];
	double bias4_2[LAYER4];
	double bias4_3[LAYER4];
	double bias4_4[LAYER4];

	
	double bias5_1[LAYER5];
	double bias5_2[LAYER5];
	double bias5_3[LAYER5];
	double bias5_4[LAYER5];

	double bias_fc1[FC1];
	double bias_fc2[FC2];
	double bias_fc3[OUTPUT];
}VGG;

typedef struct Feature
{
	double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];

	double layer1_conv1[LAYER1][LENGTH_FEATURE0][LENGTH_FEATURE0];
	double layer1_conv2[LAYER1][LENGTH_FEATURE0][LENGTH_FEATURE0];
	double layer1_pool[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];

	double layer2_conv1[LAYER2][LENGTH_FEATURE1][LENGTH_FEATURE1];
	double layer2_conv2[LAYER2][LENGTH_FEATURE1][LENGTH_FEATURE1];
	double layer2_pool[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];

	double layer3_conv1[LAYER3][LENGTH_FEATURE2][LENGTH_FEATURE2];
	double layer3_conv2[LAYER3][LENGTH_FEATURE2][LENGTH_FEATURE2];
	double layer3_conv3[LAYER3][LENGTH_FEATURE2][LENGTH_FEATURE2];
	double layer3_conv4[LAYER3][LENGTH_FEATURE2][LENGTH_FEATURE2];
	double layer3_pool[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];

	double layer4_conv1[LAYER4][LENGTH_FEATURE3][LENGTH_FEATURE3];
	double layer4_conv2[LAYER4][LENGTH_FEATURE3][LENGTH_FEATURE3];
	double layer4_conv3[LAYER4][LENGTH_FEATURE3][LENGTH_FEATURE3];
	double layer4_conv4[LAYER4][LENGTH_FEATURE3][LENGTH_FEATURE3];
	double layer4_pool[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];

	double layer5_conv1[LAYER5][LENGTH_FEATURE4][LENGTH_FEATURE4];
	double layer5_conv2[LAYER5][LENGTH_FEATURE4][LENGTH_FEATURE4];
	double layer5_conv3[LAYER5][LENGTH_FEATURE4][LENGTH_FEATURE4];
	double layer5_conv4[LAYER5][LENGTH_FEATURE4][LENGTH_FEATURE4];
	double layer5_pool[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];

	double fc1[FC1];
	double fc2[FC2];
	double output[OUTPUT];

}Feature;

void TrainBatch(VGG *vggnet, image *inputs, uint8 *labels, int batchSize);

void Train(VGG *vggnet, image input, uint8 label);

uint8 Predict(VGG *vggnet, image input, uint8 count);

void Initial(VGG *vggnet);
double relu(double x);