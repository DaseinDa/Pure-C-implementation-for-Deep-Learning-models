#pragma once

#define LENGTH_KERNEL	3
#define MAXPOOL_SIZE 2

#define PADDING 1
#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1_1	(LENGTH_FEATURE0 - LENGTH_KERNEL + 2*PADDING+1) //28-3+2+1 ->28*28
#define LENGTH_FEATURE1_2	(LENGTH_FEATURE1_1/2) //14*14

#define LENGTH_FEATURE2_1	(LENGTH_FEATURE1_2 - LENGTH_KERNEL +2*PADDING + 1) //14-3+2+1=14
#define	LENGTH_FEATURE2_2	(LENGTH_FEATURE2_1/2)  //7*7

#define LENGTH_FEATURE3_1	(LENGTH_FEATURE2_2 - LENGTH_KERNEL +2*PADDING+ 1) //7*7

#define LENGTH_FEATURE4_1	(LENGTH_FEATURE3_1 - LENGTH_KERNEL +2*PADDING+ 1)//7*7

#define MAXPOOL5_2_KERNEL 3
#define MAXPOOL5_2_STRIDE 2

#define LENGTH_FEATURE5_1	(LENGTH_FEATURE4_1 - LENGTH_KERNEL+2*PADDING+ 1)	//7*7
#define LENGTH_FEATURE5_2	((LENGTH_FEATURE5_1 - MAXPOOL5_2_KERNEL)/MAXPOOL5_2_STRIDE+ 1) //(7-3)/2 +1 = 3 no padding

#define FC1_OUTPUT	1024
#define FC2_OUTPUT	512
#define FC3_OUTPUT	10


#define INPUT			3
#define LAYER1			32
#define LAYER2			64
#define LAYER3			128
#define LAYER4			256
#define LAYER5			256

#define OUTPUT          10

#define ALPHA 0.1


typedef unsigned char uint8;
typedef uint8 image[3][32][32];



typedef struct AlexNet
{
	double weight1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];//Layer1 kernel
	double weight2[LAYER1][LAYER2][LENGTH_KERNEL][LENGTH_KERNEL];//Layer2 Kernel
	double weight3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];//Layer3 Kernel
	//double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];
	double weight4[LAYER3][LAYER4][LENGTH_KERNEL][LENGTH_KERNEL];//Layer4 Kernel
	double weight5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];//Layer5 Kernel


	double fc1[LAYER5*LENGTH_FEATURE5_2*LENGTH_FEATURE5_2][FC1_OUTPUT];
	double fc2[FC1_OUTPUT][FC2_OUTPUT];
	double fc3[FC2_OUTPUT][FC3_OUTPUT];



	double bias1[LAYER1];
	double bias2[LAYER2];
	double bias3[LAYER3];
	double bias4[LAYER4];
	double bias5[LAYER5];

	double bias_fc1[FC1_OUTPUT];
	double bias_fc2[FC2_OUTPUT];
	double bias_fc3[FC3_OUTPUT];

}AlexNet;

typedef struct Feature
{
	double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];

	double layer1_conv[LAYER1][LENGTH_FEATURE1_1][LENGTH_FEATURE1_1];
	double layer1_pool[LAYER1][LENGTH_FEATURE1_2][LENGTH_FEATURE1_2];

	double layer2_conv[LAYER2][LENGTH_FEATURE2_1][LENGTH_FEATURE2_1];
	double layer2_pool[LAYER2][LENGTH_FEATURE2_2][LENGTH_FEATURE2_2];

	double layer3_conv[LAYER3][LENGTH_FEATURE3_1][LENGTH_FEATURE3_1];
	
	double layer4_conv[LAYER4][LENGTH_FEATURE4_1][LENGTH_FEATURE4_1];


	double layer5_conv[LAYER5][LENGTH_FEATURE5_1][LENGTH_FEATURE5_1];//256*7*7
	double layer5_pool[LAYER5][LENGTH_FEATURE5_2][LENGTH_FEATURE5_2];//256*3*3

	double fc1[FC1_OUTPUT];
	double fc2[FC2_OUTPUT];
	double output[FC3_OUTPUT];

}Feature;


typedef struct Feature_Pad
{
	double input[INPUT][LENGTH_FEATURE0+2*PADDING][LENGTH_FEATURE0+2*PADDING];

	double layer1_conv[LAYER1][LENGTH_FEATURE1_1][LENGTH_FEATURE1_1];
	double layer1_pool[LAYER1][LENGTH_FEATURE1_2+2*PADDING][LENGTH_FEATURE1_2+2*PADDING];

	double layer2_conv[LAYER2][LENGTH_FEATURE2_1][LENGTH_FEATURE2_1];
	double layer2_pool[LAYER2][LENGTH_FEATURE2_2+2*PADDING][LENGTH_FEATURE2_2+2*PADDING];

	double layer3_conv[LAYER3][LENGTH_FEATURE3_1+2*PADDING][LENGTH_FEATURE3_1+2*PADDING];
	
	double layer4_conv[LAYER4][LENGTH_FEATURE4_1+2*PADDING][LENGTH_FEATURE4_1+2*PADDING];


	double layer5_conv[LAYER5][LENGTH_FEATURE5_1][LENGTH_FEATURE5_1];//256*7*7
	double layer5_pool[LAYER5][LENGTH_FEATURE5_2][LENGTH_FEATURE5_2];//256*3*3

	double fc1[FC1_OUTPUT];
	double fc2[FC2_OUTPUT];
	double output[FC3_OUTPUT];

}Feature_Pad;

void TrainBatch(AlexNet *alexnet, image *inputs, uint8 *labels, int batchSize);

void Train(AlexNet *alexnet, image input, uint8 label);

uint8 Predict(AlexNet *alexnet, image input, uint8 count);

void Initial(AlexNet *alexnet);

