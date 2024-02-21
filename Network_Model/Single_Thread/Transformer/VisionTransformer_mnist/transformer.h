#pragma once
#define PI 3.1415926
#define LENGTH_KERNEL	5
#define MAXPOOL_SIZE 2

#define PADDING 1
#define LENGTH_FEATURE0	28
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

#define FC	512


#define INPUT			1
#define LAYER1			16
#define LAYER2			32

#define OUTPUT          10

#define ALPHA 0.01
#define RES1_CHANNEL 16
#define RES2_CHANNEL 32

#define RES_LENGTH_KERNEL 3

#define INPUTCHANNEL 1
#define PATCH_SIZE 7
#define PATCH_NUM  (28/PATCH_SIZE)*(28/PATCH_SIZE)
#define PATCH_DIM  INPUTCHANNEL*PATCH_SIZE*PATCH_SIZE
#define DIM 64
#define DEPTH 6
#define HEADER 8
#define MLP_DIM 128

typedef unsigned char uint8;
typedef uint8 image[28][28];

typedef struct ResBlock1
{
	double weight1[RES1_CHANNEL][RES1_CHANNEL][RES_LENGTH_KERNEL][RES_LENGTH_KERNEL];
	double weight2[RES1_CHANNEL][RES1_CHANNEL][RES_LENGTH_KERNEL][RES_LENGTH_KERNEL];
	double bias1[RES1_CHANNEL];
	double bias2[RES1_CHANNEL];	
}ResBlock1;


typedef struct ResBlock2
{
	double weight1[RES2_CHANNEL][RES2_CHANNEL][RES_LENGTH_KERNEL][RES_LENGTH_KERNEL];
	double weight2[RES2_CHANNEL][RES2_CHANNEL][RES_LENGTH_KERNEL][RES_LENGTH_KERNEL];	
	double bias1[RES1_CHANNEL];
	double bias2[RES1_CHANNEL];	
}ResBlock2;	


typedef struct Res1_Feature
{
	double input_pad[RES1_CHANNEL][14][14];
	double conv1[RES1_CHANNEL][12][12];

	double conv1_pad[RES1_CHANNEL][14][14];
	double conv2[RES1_CHANNEL][12][12];
}Res1_Feature;

typedef struct Res2_Feature
{
	double input_pad[RES2_CHANNEL][6][6];
	double conv1[RES2_CHANNEL][4][4];
	double conv1_pad[RES2_CHANNEL][6][6];
	double conv2[RES2_CHANNEL][4][4];
}Res2_Feature;



typedef struct ResNet
{
	double weight1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];//Layer1 kernel
	//maxpool
	ResBlock1 res1;
	double weight2[LAYER1][LAYER2][LENGTH_KERNEL][LENGTH_KERNEL];//Layer2 Kernel
	//maxpool
	ResBlock2 res2;

	double fc[512][OUTPUT];


	double bias1[LAYER1];
	double bias2[LAYER2];


	double bias_fc[OUTPUT];

}ResNet;




typedef struct Feature
{
	double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
	double conv1[LAYER1][24][24];
	double max1[LAYER1][12][12];

	Res1_Feature res1f;

	double conv2[LAYER2][8][8];
	double max2[LAYER2][4][4];
	Res2_Feature res2f;

	double output[10];

}Feature;


typedef struct ViT
{
	double pos_embedding[PATCH_NUM+1][DIM];//每个patch的pos_embedding是固定的,通过参数梯度更新
	double patch_embedding_weight[INPUT][PATCH_DIM][DIM];//线性层相当于全连接层，每个patch使用相同的linear
	double cls_token[1][DIM];//用来和patch_embedding拼接
	double patch_bias[DIM];

	double q1x[HEADER][DIM][DIM];
	double k1x[HEADER][DIM][DIM];
	double v1x[HEADER][DIM][DIM];
	double out_weight1[DIM][DIM];
	double feed11[DIM][MLP_DIM];
	double feed12[MLP_DIM][DIM];
	
	double q2x[HEADER][DIM][DIM];
	double k2x[HEADER][DIM][DIM];
	double v2x[HEADER][DIM][DIM];
	double out_weight2[DIM][DIM];
	double feed21[DIM][MLP_DIM];
	double feed22[MLP_DIM][DIM];

	
	double q3x[HEADER][DIM][DIM];
	double k3x[HEADER][DIM][DIM];
	double v3x[HEADER][DIM][DIM];
	double out_weight3[DIM][DIM];
	double feed31[DIM][MLP_DIM];
	double feed32[MLP_DIM][DIM];

	
	double q4x[HEADER][DIM][DIM];
	double k4x[HEADER][DIM][DIM];
	double v4x[HEADER][DIM][DIM];
	double out_weight4[DIM][DIM];
	double feed41[DIM][MLP_DIM];
	double feed42[MLP_DIM][DIM];
	
	double q5x[HEADER][DIM][DIM];
	double k5x[HEADER][DIM][DIM];
	double v5x[HEADER][DIM][DIM];
	double out_weight5[DIM][DIM];
	double feed51[DIM][MLP_DIM];
	double feed52[MLP_DIM][DIM];
	
	double q6x[HEADER][DIM][DIM];
	double k6x[HEADER][DIM][DIM];
	double v6x[HEADER][DIM][DIM];
	double out_weight6[DIM][DIM];
	double feed61[DIM][MLP_DIM];
	double feed62[MLP_DIM][DIM];

	double mlp_w1[DIM][MLP_DIM];
	double mlp_w2[MLP_DIM][OUTPUT];
}ViT;


typedef struct ViTFeature
{
	double input[INPUT][28][28];
	double input_patch[PATCH_NUM][INPUT][PATCH_SIZE][PATCH_SIZE];
	double patch_embedding[PATCH_NUM][DIM];
	double patch_embedding_cls[PATCH_NUM+1][DIM];
	double q1[HEADER][PATCH_NUM+1][DIM];
	double k1[HEADER][PATCH_NUM+1][DIM];
	double v1[HEADER][PATCH_NUM+1][DIM];
	double z1[HEADER][PATCH_NUM+1][DIM];
	double attention_out1[PATCH_NUM+1][DIM];
	double out1[PATCH_NUM+1][DIM];
	double trans_feed11[PATCH_NUM+1][MLP_DIM];
	double trans_feed12[PATCH_NUM+1][DIM];
	
	double q2[HEADER][PATCH_NUM+1][DIM];
	double k2[HEADER][PATCH_NUM+1][DIM];
	double v2[HEADER][PATCH_NUM+1][DIM];
	double z2[HEADER][PATCH_NUM+1][DIM];
	double attention_out2[PATCH_NUM+1][DIM];
	double out2[PATCH_NUM+1][DIM];
	double trans_feed21[PATCH_NUM+1][MLP_DIM];
	double trans_feed22[PATCH_NUM+1][DIM];


	double q3[HEADER][PATCH_NUM+1][DIM];
	double k3[HEADER][PATCH_NUM+1][DIM];
	double v3[HEADER][PATCH_NUM+1][DIM];
	double z3[HEADER][PATCH_NUM+1][DIM];
	double attention_out3[PATCH_NUM+1][DIM];
	double out3[PATCH_NUM+1][DIM];
	double trans_feed31[PATCH_NUM+1][MLP_DIM];
	double trans_feed32[PATCH_NUM+1][DIM];


	double q4[HEADER][PATCH_NUM+1][DIM];
	double k4[HEADER][PATCH_NUM+1][DIM];
	double v4[HEADER][PATCH_NUM+1][DIM];
	double z4[HEADER][PATCH_NUM+1][DIM];
	double attention_out4[PATCH_NUM+1][DIM];
	double out4[PATCH_NUM+1][DIM];
	double trans_feed41[PATCH_NUM+1][MLP_DIM];
	double trans_feed42[PATCH_NUM+1][DIM];

	double q5[HEADER][PATCH_NUM+1][DIM];
	double k5[HEADER][PATCH_NUM+1][DIM];
	double v5[HEADER][PATCH_NUM+1][DIM];
	double z5[HEADER][PATCH_NUM+1][DIM];
	double attention_out5[PATCH_NUM+1][DIM];
	double out5[PATCH_NUM+1][DIM];
	double trans_feed51[PATCH_NUM+1][MLP_DIM];
	double trans_feed52[PATCH_NUM+1][DIM];

	double q6[HEADER][PATCH_NUM+1][DIM];
	double k6[HEADER][PATCH_NUM+1][DIM];
	double v6[HEADER][PATCH_NUM+1][DIM];
	double z6[HEADER][PATCH_NUM+1][DIM];
	double attention_out6[PATCH_NUM+1][DIM];
	double out6[PATCH_NUM+1][DIM];
	double trans_feed61[PATCH_NUM+1][MLP_DIM];
	double trans_feed62[PATCH_NUM+1][DIM];

	double mlp1[MLP_DIM];
	double mlp2[OUTPUT];
	double output[OUTPUT];

}ViTFeature;




void TrainBatch(ViT *vit, image *inputs, uint8 *labels, int batchSize);

void Train(ViT *vit, image input, uint8 label);

uint8 Predict(ViT *vit, image input, uint8 count);

void Initial(ViT *vit);

