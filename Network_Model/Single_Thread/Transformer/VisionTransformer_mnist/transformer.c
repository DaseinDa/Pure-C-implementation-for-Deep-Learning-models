#include "transformer.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>

 #define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

 #define GETCOUNT(array)  (sizeof(array)/sizeof(double))

 #define FOREACH(i,count) for (int i = 0; i < count; ++i)


// static Feature_Pad features_pad = { 0 };
// static Feature_Pad errors_pad = { 0 };
static ViTFeature features = { 0 };
static ViTFeature errors = { 0 };

static ViT vit={0};
static ViT deltas={0};
// Feature* features;
// Feature* errors;
// AlexNet* deltas;






#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}

#define Matrix_FC(input,output,weight)											\	
{																	\		
	FOREACH(o0,GETLENGTH(input))			\
		FOREACH(o1, GETLENGTH(*input))				\
			FOREACH(o2, GETLENGTH(**input))								\	
				FOREACH(i0, GETLENGTH(output))	\
				(output[i0] += input[o0][o1][o2]* weight[o0*o1+o2][i0]);					\	
}																						\															


#define FC(input,output,weight)\
{\
	FOREACH(o0,GETLENGTH(input))\
		FOREACH(i0,GETLENGTH(output))\
			output[i0] += weight[o0][i0]*input[o0];\
}\

// #define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
// {																			\
// 	for (int x = 0; x < GETLENGTH(weight); ++x)								\
// 		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
// 			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
// 	FOREACH(i, GETCOUNT(inerror))											\
// 		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
// 	FOREACH(j, GETLENGTH(outerror))											\
// 		FOREACH(i, GETCOUNT(outerror[j]))									\
// 		bd[j] += ((double *)outerror[j])[i];								\
// 	for (int x = 0; x < GETLENGTH(weight); ++x)								\
// 		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
// 			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
// }


#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}




#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}
//len0是输入矩阵的行数长度除以输出矩阵的行数长度
//len1是输入矩阵的列数宽度除以输出矩阵的列数宽度
//len0和len1是池化核的大小
#define SUBSAMP_MAX_STRIDE_FORWARD(input,output,kernel_size,stride)								\
{																								\
	const int len0 = kernel_size;																\
	const int len1 = kernel_size;																\
	const int s = stride;																		\
	FOREACH(i,GETLENGTH(output))																\
		FOREACH(o0,GETLENGTH(*(output)))														\
			FOREACH(o1,GETLENGTH(**(output)))													\
				{																				\
					int x0=0, x1=0, ismax;														\
					FOREACH(l0,len0)															\
						FOREACH(l1,len1)														\
						{																		\
							ismax = input[i][o0*s + l0][o1*s + l1] > input[i][o0*s+ x0][o1*s + x1];\
							x0 += ismax * (l0 - x0);											\
							x1 += ismax * (l1 - x1);											\
						}																		\
					output[i][o0][o1] = input[i][o0*s + x0][o1*s + x1];							\
				}																				\
}																								\

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}


#define SUBSAMP_MAX_STRIDE_BACKWARD(input,inerror,outerror,kernel_size, stride)										\
{																								\
	const int len0 = kernel_size;																\
	const int len1 = kernel_size;																\
	const int s = stride;																		\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*s + l0][o1*s + l1] > input[i][o0*s + x0][o1*s + x1];			\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*s + x0][o1*s + x1] = outerror[i][o0][o1];									\
	}																							\
}																								\


#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y]; 	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}
// ((double *)output)[j] = action(((double *)output)[j] + bias[j]);全连接计算后的激活函数
// ((double *)output)[y] += ((double *)input)[x] * weight[x][y];全连接计算
//input是三维的， output是一维的



// #define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
// {																			\
// 	for (int x = 0; x < GETLENGTH(weight); ++x)									\
// 		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
// 			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
// 			printf("1111111111111111111\n");									\
// 	FOREACH(i, GETCOUNT(inerror))												\
// 		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
// 		printf("2222222222222222222\n");									\
// 	FOREACH(j, GETLENGTH(outerror))												\
// 		bd[j] += ((double *)outerror)[j];										\
// 		printf("333333333333333333333333\n");									\
// 	for (int x = 0; x < GETLENGTH(weight); ++x)									\
// 		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
// 			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
// 			printf("44444444444444444444\n");									\
// }


#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}


#define PADDING_fill(input_pad, input)											\
{																				\
	FOREACH(i,GETLENGTH(input))													\
		FOREACH(j,GETLENGTH(*input))											\
			FOREACH(k,GETLENGTH(*input))										\
				input_pad[i][j+PADDING][k+PADDING]=input[i][j][k];				\		
}																				\

#define PADDING_remove(input, input_pad)											\
{																				\
	FOREACH(i,GETLENGTH(input))													\
		FOREACH(j,GETLENGTH(*input))											\
			FOREACH(k,GETLENGTH(*input))										\
				input[i][j][k]=input_pad[i][j+PADDING][k+PADDING];				\		
}	

#define CONVOLUTION_RES_FORWARD(res_input,input,output,weight,bias,action)		\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(i, GETCOUNT(output))												\
		((double*)output)[i] += ((double*)res_input)[i];						\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}
#define ResBlock_Forward(input,output,res,action)								\
{																				\
	PADDING_fill(output.input_pad,input);										\
	CONVOLUTION_FORWARD(output.input_pad,output.conv1,res.weight1,res.bias1,action);\
	PADDING_fill(output.conv1_pad,output.conv1);								\
	CONVOLUTION_RES_FORWARD(input, output.conv1_pad,output.conv2,res.weight2,res.bias2,action);\
}																					\


#define ResBlock_Backward(res,res_error,res_weight,res_deltas,actiongrad)	\
{																			\
	CONVOLUTION_BACKWARD(res.conv1_pad, res_error.conv1_pad,res_error.conv2,res_weight.weight2,res_deltas.weight2,res_deltas.bias2,actiongrad);\
	PADDING_remove(res_error.conv1,res_error.conv1_pad);											\
	CONVOLUTION_BACKWARD(res.input_pad, res_error.input_pad,res_error.conv1,res_weight.weight1,res_deltas.weight1,res_deltas.bias1,actiongrad);\
}
// #define PATCH_DOT_PRODUCT_FORWARD(input,output,weight,bias)					\
// {																			\
// 	for (int x = 0; x < GETLENGTH(weight); ++x)								\
// 		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
// 			((double *)output)[y] += ((double *)input)[x] * weight[x][y]; 	\
// }


#define Feature2Patch(input,output)											\
{																			\
	size_t image_size = GETLENGTH(*input);									\	
	size_t patch_size=GETLENGTH(**output);									\
	size_t patch_num_row = image_size/patch_size;							\		
	FOREACH(o0, GETLENGTH(output))											\
		FOREACH(o1, GETLENGTH(output))										\
			FOREACH(o2, GETLENGTH(output))									\
				FOREACH(o3, GETLENGTH(output))								\
					output[o0][o1][o2][o3]=input[o1][o0/patch_num_row*patch_size+o2][o0%patch_num_row*patch_size+o3];\
}
//patch_num_row 一行有多少patch等于一列有多少patch

#define DOT_PATCH_EMBEDDING_PRODUCT_FORWARD(input,output,weight,bias)		\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y]; 	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] += bias[j];									\
}

#define PATCH_EMBEDDING_FROWARD(input_patch,output,weight,bias)						\
{																					\
	FOREACH(o0,GETLENGTH(input_patch))												\
		FOREACH(o1,GETLENGTH(*input_patch))											\
			DOT_PATCH_EMBEDDING_PRODUCT_FORWARD(input_patch[o0][o1],output[o0],weight[o1],bias);\
}					
#define PATCH_CLS(input,output)																\
{																							\
	size_t cls_dim = GETLENGTH(input);														\
	FOREACH(o0,GETLENGTH(*output))															\
		input[cls_dim-1][o0]=output[cls_dim][o0];											\
}	

#define PATCH_POS_PLUS(input,output)																	\
{																										\
	FOREACH(o0,GETLENGTH(output))																		\
		FOREACH(o1, GETLENGTH(*output))																	\
			output[o0][o1]+=input[o0][o1];																\
}
#define DOT_ATTENTION_FORWARD(input,output,weight)							\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y]; 	\
}

#define Matrix_Mul_Tran(x,y,z)												\
{																			\
	FOREACH(i,GETLENGTH(z))													\
		FOREACH(j,GETLENGTH(z))												\
			FOREACH(k,GETLENGTH(*x))										\
				z[i][j]+=x[i][k]*y[j][k];									\
}

#define Matrix_Mul_NoNeedTran(x,y,z)										\
{																			\
	FOREACH(i,GETLENGTH(z))													\
		FOREACH(j,GETLENGTH(z))												\
			FOREACH(k,GETLENGTH(*x))										\
				z[i][j]+=x[i][k]*y[k][j];									\
}
#define SOFTMAX(input)														\
{																			\
	double inner = 0;														\
	double res = 0;															\
	for (int j = 0; j < GETCOUNT(input); ++j)								\
		{																	\
			res += exp(((double *)input)[j]);								\
		}																	\
	for (int i = 0; i < GETCOUNT(input); ++i)								\
	{																		\
		((double *)input)[i]/=res;											\
	}																		\
}
#define FeedForward(input,out_feed1,out_feed2,weight1,weight2)				\
{																			\
	FOREACH(o0,GETLENGTH(input))											\
		DOT_ATTENTION_FORWARD(input[o0],out_feed1[o0],weight1);				\
	FOREACH(o0,GETLENGTH(out_feed1))										\
		DOT_ATTENTION_FORWARD(out_feed1[o0],out_feed2[o0],weight2);			\
	FOREACH(j, GETCOUNT(out_feed1))											\
		((double *)out_feed1)[j] = gelu(((double *)out_feed1)[j] );			\
	FOREACH(i, GETCOUNT(out_feed2))											\
			((double*)out_feed2)[i] += ((double*)input)[i];					\
}

#define SELF_ATTENTION(input,k,q,v,z,att_out,out,qx,kx,vx,out_w)													\
{																											\
		size_t dim=GETLENGTH(*input);																		\
		double scale=invSqrt(dim);																			\
		for(int o0=0;o0<GETLENGTH(input);o0++)																\
		{																									\
			for(int o1=0;o1<GETLENGTH(q);o1++)																\
			{																								\
			DOT_ATTENTION_FORWARD(input[o0],q[o1][o0],qx[o1]);												\
			DOT_ATTENTION_FORWARD(input[o0],k[o1][o0],kx[o1]);												\
			DOT_ATTENTION_FORWARD(input[o0],v[o1][o0],vx[o1]); 												\	
			}																								\
		}																									\
		double qk[HEADER][PATCH_NUM+1][PATCH_NUM+1];														\
		FOREACH(num,HEADER)																					\
			Matrix_Mul_Tran(q[num],k[num],qk[num]);															\
		SOFTMAX(qk);																						\
		FOREACH(num,HEADER)																					\
			Matrix_Mul_NoNeedTran(qk[num],v[num],z[num]);													\
		FOREACH(o0,GETLENGTH(z))																			\
			FOREACH(o1,GETLENGTH(*z))																		\
				FOREACH(o2,GETLENGTH(**z))																	\
					att_out[o1][o2]+=z[o0][o1][o2];															\
		DOT_ATTENTION_FORWARD(att_out,out,out_w);															\
}
#define Transformer(input,k,q,v,z,att_out,out,qx,kx,vx,out_w)			\
{																		\
		SELF_ATTENTION(input,k,q,v,z,att_out,out,qx,kx,vx,out_w);		\
		FOREACH(i, GETCOUNT(out))										\
			((double*)out)[i] += ((double*)input)[i];					\
}

#define MLP(input,out1,out2,weight1,weight2)								\
{																			\
	DOT_ATTENTION_FORWARD(input,out1,weight1);								\
	FOREACH(j, GETCOUNT(out1))												\
		((double *)out1)[j] = gelu(((double *)out1)[j] );					\
	DOT_ATTENTION_FORWARD(out1,out2,weight2);								\
}


double invSqrt(double x)
{
    int i = *(int *)&x;
    i = 1597463007-(i>>1);
    x = *(float*)&i;
    return x;
}
double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

double gelu(double x)
{
	return 0.5*x*(1+tanh(sqrt(2/PI)*(x+0.044715*(pow(x,3)))));
}

double gelugrad(double x)
{
	double cdf=0.5*(1+tanh(sqrt(2/PI)*(x+0.044715*(pow(x,3)))));
	double pdf=(1/(sqrt(2*PI)))* exp(-(x*x)/2);
	return cdf+x*pdf;
}

static void forward(ViT *vit, double(*action)(double))
{
	Feature2Patch(features.input, features.input_patch);
	PATCH_EMBEDDING_FROWARD(features.input_patch,features.patch_embedding,vit->patch_embedding_weight,vit->patch_bias);
	PATCH_CLS(features.patch_embedding,features.patch_embedding_cls);
	PATCH_POS_PLUS(vit->pos_embedding,features.patch_embedding_cls);
	Transformer(features.patch_embedding_cls,features.q1,features.k1,features.v1,features.z1,features.attention_out1,features.out1,vit->q1x,vit->k1x,vit->v1x,vit->out_weight1);
	FeedForward(features.out1,features.trans_feed11,features.trans_feed12,vit->feed11,vit->feed12);

	Transformer(features.trans_feed12,features.q2,features.k2,features.v2,features.z2,features.attention_out2,features.out2,vit->q2x,vit->k2x,vit->v2x,vit->out_weight2);
	FeedForward(features.out2,features.trans_feed21,features.trans_feed22,vit->feed21,vit->feed22);

	Transformer(features.trans_feed22,features.q3,features.k3,features.v3,features.z3,features.attention_out3,features.out3,vit->q3x,vit->k3x,vit->v3x,vit->out_weight3);
	FeedForward(features.out3,features.trans_feed31,features.trans_feed32,vit->feed31,vit->feed32);

	Transformer(features.trans_feed32,features.q4,features.k4,features.v4,features.z4,features.attention_out4,features.out4,vit->q4x,vit->k4x,vit->v4x,vit->out_weight4);
	FeedForward(features.out4,features.trans_feed41,features.trans_feed42,vit->feed41,vit->feed42);

	Transformer(features.trans_feed42,features.q5,features.k5,features.v5,features.z5,features.attention_out5,features.out5,vit->q5x,vit->k5x,vit->v5x,vit->out_weight5);
	FeedForward(features.out5,features.trans_feed51,features.trans_feed52,vit->feed51,vit->feed52);

	Transformer(features.trans_feed52,features.q6,features.k6,features.v6,features.z6,features.attention_out6,features.out6,vit->q6x,vit->k6x,vit->v6x,vit->out_weight6);
	FeedForward(features.out6,features.trans_feed61,features.trans_feed62,vit->feed61,vit->feed62);

	MLP(features.trans_feed62[PATCH_NUM],features.mlp1,features.mlp2,vit->mlp_w1,vit->mlp_w2);

}

static void backward(ViT *vit, double(*actiongrad)(double))
{


}

static inline void load_input(image input)
{
	//double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = &features.input;
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		//features.input[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
		features.input[0][j][k] = (input[j][k] - mean) / std;
		//printf("The calculate is %f\n",(input[j][k] - mean) / std);
	}
	// printf("The getlength of features.input is %d\n",GETLENGTH(features.input));
	// printf("The getlength of *features.input is %d\n",GETLENGTH(*features.input));
	// printf("The getlength of **features.input is %d\n",GETLENGTH(**features.input));

	// FOREACH(j, GETLENGTH(*features.input))
	// 	FOREACH(k,  GETLENGTH(**features.input))
	// {
	// 	printf("The features after load is %f\n",features.input[0][j][k]);
	// }
}



static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
	double inner = 0;
	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res; //这里是softmax的输出值 
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];//inner 是正确结果在模型中的预测概率值
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;// 如果j=i,loss=(1-pi-pi)*pi, 如果j不等于i, loss=(-pj-pi)pj这是梯度
	}
}

static void load_target(int label)
{
	double *output = (double *)features.output;
	double *error = (double *)errors.output;
	softmax(output, error, label, GETCOUNT(features.output));//计算softmax梯度的损失函数
}

static uint8 get_result(ViTFeature *features, uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;//预测结果输出值中的最大值,softmax主要是梯度传导用的
		}
	}
	return result;
}

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}


void TrainBatch(ViT *vit, image *inputs, uint8 *labels, int batchSize)
{
	 //double buffer[GETCOUNT(AlexNet)] = { 0 };
	 //double *buffer = (double *)malloc(sizeof(AlexNet));
	 int i = 0;
//#pragma omp parallel for

		int right=0;
		memset(&deltas, 0, sizeof(ViT));
		
		//printf("%f\n",((double*)&deltas)[0]);
		
	for (i = 0; i < batchSize; ++i)
	{
		printf("Here is TrainBatch %d\n",i);
		//Feature features = { 0 };
		//Feature errors = { 0 };
		memset(&features, 0, sizeof(ViTFeature));
		memset(&errors, 0, sizeof(ViTFeature));
		// memset(&features_pad, 0, sizeof(Feature_Pad));
		// memset(&errors_pad, 0, sizeof(Feature_Pad));



		load_input(inputs[i]);


		// FOREACH(i,GETLENGTH(resnet->weight1))
		// 	FOREACH(j,GETLENGTH(*resnet->weight1))
		// 		FOREACH(k,GETLENGTH(**resnet->weight1))
		// 			FOREACH(p,GETLENGTH(***resnet->weight1))
		// 				printf("##%f\n",resnet->weight1[i][j][k][p]);
		// sleep(2);
		
		forward(vit,  gelu);
		printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
		load_target(labels[i]);
		printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
		backward(vit,gelugrad);
		//  #pragma omp critical
		// {
		// 	FOREACH(j, GETCOUNT(AlexNet))
		// 		buffer[j] += ((double *)&deltas)[j];
		// }
			// FOREACH(j, GETCOUNT(AlexNet))
		 	// 	buffer[j] += ((double *)&deltas)[j];
	// right+=(get_result(&features,10) == labels[i]);
	// printf("right:%d\n", right);
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(ViT))
			printf("%f   %f    %f\n",((double *)vit)[i], ((double *)&deltas)[i]*ALPHA,((double *)vit)[i] +((double *)&deltas)[i]/300);
	sleep(2);
	FOREACH(i, GETCOUNT(ViT))
		((double *)vit)[i] += ALPHA * ((double *)&deltas)[i]/300;
}

// void Train(LeNet5 *lenet, image input, uint8 label)
// {
// 	Feature features = { 0 };
// 	Feature errors = { 0 };
// 	LeNet5 deltas = { 0 };
// 	load_input(&features, input);
// 	forward(lenet, &features, relu);
// 	load_target(&features, &errors, label);
// 	backward(lenet, &deltas, &errors, &features, relugrad);
// 	FOREACH(i, GETCOUNT(LeNet5))
// 		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
// }

uint8 Predict(ViT *vit, image input,uint8 count)
{
	load_input(input);
	forward(vit, relu);
	return get_result(&features, count);
}

void Initial(ViT *vit)
{
	for (double *pos = (double *)vit->pos_embedding; pos < (double *)vit->cls_token+DIM; *pos++ = f64rand());

}
