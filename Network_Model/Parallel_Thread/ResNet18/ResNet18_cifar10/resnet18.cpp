#include "resnet18.h"
#include <memory.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <fstream>
#include <chrono>
#include <iostream>
using namespace std;
#define OMP_PARA_INTERNAL _Pragma("omp parallel for")
decltype(std::chrono::system_clock::now()) start_res18;
decltype(std::chrono::system_clock::now() - std::chrono::system_clock::now()) duration_res18;
void start_timer_res18() {
	start_res18 = std::chrono::system_clock::now();
}
void stop_timer_res18() {
	duration_res18 = std::chrono::system_clock::now() - start_res18;
}

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)
#define ToPad(input_pad,input,PADDING){													\
	OMP_PARA_INTERNAL																	\
	for(int i=0;i<GETLENGTH(input);i++){												\
		for(int o0=0;i<GETLENGTH(*input);i++){											\
			memcpy(input_pad[i][o0+PADDING]+PADDING,input[i][o0],sizeof(input[0][0]));\
	}}																					\
}																						\

#define extractPad(input, input_pad,PADDING){											\
	for(int i=0;i<GETLENGTH(input);i++){												\
		for(int o0=0;i<GETLENGTH(*input);i++){											\
			memcpy(input[i][o0],input_pad[i][o0+PADDING]+PADDING,sizeof(input[0][0]));\
	}}																					\
}		
#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	OMP_PARA_INTERNAL																	\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}


#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	OMP_PARA_INTERNAL																	\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}
#define CONVOLUTION_PADDING_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	double input_pad[GETLENGTH(input)][GETLENGTH(*input)+2*PADDING][GETLENGTH(**input)+2*PADDING]={0};	\
	double inerror_pad[GETLENGTH(input)][GETLENGTH(*input)+2*PADDING][GETLENGTH(**input)+2*PADDING]={0};\
	ToPad(input_pad,input,PADDING);													\
	ToPad(inerror_pad,inerror,PADDING);												\
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror_pad[x], weight[x][y]);			\
	OMP_PARA_INTERNAL																	\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input_pad)[i]);			\
	OMP_PARA_INTERNAL																	\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input_pad[x], wd[x][y], outerror[y]);\
	extractPad(input,input_pad,PADDING);										\
	extractPad(inerror,inerror_pad,PADDING);										\
}
#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	OMP_PARA_INTERNAL																	\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}
#define CONVOLUTE_VALID_1D(input,output,weight)											\
{																						\
	OMP_PARA_INTERNAL																	\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0*2 + w0][o1*2 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTION_FORWARD_1D(input,output,weight,action)						\
{																				\
	OMP_PARA_INTERNAL															\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID_1D(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i]);	\
}
#define CONVOLUTION_BACKWARD_1D(input,inerror,outerror,weight,wd,action)						\
{																				\
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL_1D(outerror[y], inerror[x], weight[x][y]);			\
	OMP_PARA_INTERNAL																	\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_BACKVALID_1D(input[x], wd[x][y], outerror[y]);		\		
}
#define CONVOLUTE_BACKVALID_1D(input,output,weight)											\
{																						\
	OMP_PARA_INTERNAL																	\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0*2][o1 + w1*2] * (weight)[w0][w1];	\
}

#define CONVOLUTE_FULL_1D(input,output,weight)												\
{																						\
	OMP_PARA_INTERNAL																	\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0*2 + w0][i1*2 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}
//stride=2
#define CONVOLUTE_STRIDE_VALID(input,output,weight,stride)								\
{																						\
	OMP_PARA_INTERNAL																	\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0*stride + w0][o1*stride + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTION_STRIDE_FORWARD(input,output,weight,bias,action)				\
{																				\
	double input_pad[GETLENGTH(input)][GETLENGTH(*input)+2*PADDING][GETLENGTH(**input)+2*PADDING]={0};	\
	ToPad(input_pad,input,PADDING);													\
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_STRIDE_VALID(input_pad[x], output[y], weight[x][y],2);		\
	OMP_PARA_INTERNAL																	\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
	extractPad(input,input_pad,PADDING);										\
}
#define CONVOLUTE_BACK_STRIDE_FULL(input,output,weight,stride)							\
{																						\
	OMP_PARA_INTERNAL																	\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0*stride + w0][i1*stride + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}
#define CONVOLUTE_BACK_STRIDE_VALID(input,output,weight,stride)							\
{																						\
	OMP_PARA_INTERNAL																	\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0*stride][o1 + w1*stride] * (weight)[w0][w1];	\
}


#define CONVOLUTION_STRIDEPADDING_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	double input_pad[GETLENGTH(input)][GETLENGTH(*input)+2*PADDING][GETLENGTH(**input)+2*PADDING]={0};	\
	double inerror_pad[GETLENGTH(input)][GETLENGTH(*input)+2*PADDING][GETLENGTH(**input)+2*PADDING]={0};\
	ToPad(input_pad,input,PADDING);													\
	ToPad(inerror_pad,inerror,PADDING);												\
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_BACK_STRIDE_FULL(outerror[y], inerror_pad[x], weight[x][y],2);\
	OMP_PARA_INTERNAL																	\
	FOREACH(i, GETCOUNT(inerror_pad))											\
		((double *)inerror_pad)[i] *= actiongrad(((double *)input_pad)[i]);			\
	OMP_PARA_INTERNAL																	\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_BACK_STRIDE_VALID(input_pad[x], wd[x][y], outerror[y],2);	\
	extractPad(input,input_pad,PADDING);										\
	extractPad(inerror,inerror_pad,PADDING);										\
}
#define CONVOLUTION_STRIDE_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_BACK_STRIDE_FULL(outerror[y], inerror[x], weight[x][y],2);\
	OMP_PARA_INTERNAL																	\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	OMP_PARA_INTERNAL																	\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_BACK_STRIDE_VALID(input[x], wd[x][y], outerror[y]);	\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	OMP_PARA_INTERNAL														\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	OMP_PARA_INTERNAL														\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	OMP_PARA_INTERNAL														\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	OMP_PARA_INTERNAL														\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}
#define CONVOLUTION_PADDING_FORWARD(input,output,weight,bias,action)						\
{																					\
	double input_pad[GETLENGTH(input)][GETLENGTH(*input)+2*PADDING][GETLENGTH(**input)+2*PADDING]={0};	\
	ToPad(input_pad,input,PADDING);													\
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input_pad[x], output[y], weight[x][y]);				\
	OMP_PARA_INTERNAL																	\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
	extractPad(input,input_pad,PADDING);										\
}

#define ResBlock1_Forward(input,output,weight,action)									\
{																						\
	CONVOLUTION_PADDING_FORWARD(input,output.conv1_1,weight.weight1_1,weight.bias1_1,action);\
	CONVOLUTION_PADDING_FORWARD(output.conv1_1, output.conv1_2,weight.weight1_2,weight.bias1_2,action);\
	OMP_PARA_INTERNAL																	\
	FOREACH(i, GETCOUNT(output.conv1_2))														\
		((double*)output.conv1_2)[i] += ((double*)input)[i];									\
	CONVOLUTION_PADDING_FORWARD(output.conv1_2,output.conv2_1,weight.weight2_1,weight.bias2_1,action);\
	CONVOLUTION_PADDING_FORWARD(output.conv2_1, output.conv2_2,weight.weight2_2,weight.bias2_2,action);\
	OMP_PARA_INTERNAL																	\
	FOREACH(i, GETCOUNT(output.conv2_2))														\
		((double*)output.conv2_2)[i] += ((double*)output.conv1_2)[i];							\
}

#define ResBlock1_Backward(feature,error,weight,deltas,action,res,reserror)									\
{																									\
	OMP_PARA_INTERNAL																	\
	FOREACH(i, GETCOUNT(error.conv2_2))													\
		((double *)error.conv1_2)[i] += ((double *)error.conv2_2)[i];								\
	CONVOLUTION_PADDING_BACKWARD(feature.conv2_1,error.conv2_1,error.conv2_2,weight.weight2_2,deltas.weight2_2,deltas.bias2_2,action);\
	CONVOLUTION_PADDING_BACKWARD(feature.conv1_2,error.conv1_2,error.conv2_1,weight.weight2_1,deltas.weight2_1,deltas.bias2_1,action);\
	OMP_PARA_INTERNAL																	\
		FOREACH(i, GETCOUNT(error.conv1_2))													\
		((double *)res)[i] += ((double *)error.conv1_2)[i];								\
	CONVOLUTION_PADDING_BACKWARD(feature.conv1_1,error.conv1_1,error.conv1_2,weight.weight1_2,deltas.weight1_2,deltas.bias1_2,action);\
	CONVOLUTION_PADDING_BACKWARD(res,reserror,error.conv1_1,weight.weight1_1,deltas.weight1_1,deltas.bias1_1,action);\
}
#define ResBlock2_Backward_FromPool(feature, error,weight,deltas,action,res,reserror)		\
{																							\
	OMP_PARA_INTERNAL																	\
	FOREACH(i, GETCOUNT(error.conv2_2))													\
		((double *)error.conv1_2)[i] += ((double *)error.conv2_2)[i];								\
	CONVOLUTION_PADDING_BACKWARD(feature.conv2_1,error.conv2_1,error.conv2_2,weight.weight2_2,deltas.weight2_2,deltas.bias2_2,action);\
	CONVOLUTION_PADDING_BACKWARD(feature.conv1_2,error.conv1_2,error.conv2_1,weight.weight2_1,deltas.weight2_1,deltas.bias2_1,action);\
	CONVOLUTION_FORWARD_1D(error.conv1_2,res,weight.conv1,action);											\
	CONVOLUTION_BACKWARD_1D(res,reserror,error.conv2_1,weight.conv1,deltas.conv1,action);											\
	CONVOLUTION_PADDING_BACKWARD(feature.conv1_1,error.conv1_1,error.conv1_2,weight.weight1_2,deltas.weight1_2,deltas.bias1_2,action);\
	CONVOLUTION_STRIDEPADDING_BACKWARD(res,reserror,error.conv1_1,weight.weight1_1,deltas.weight1_1,deltas.bias1_1,actiongrad)\
}

#define ResBlock2_Forward(input,output,weight,action)											\
{																								\
	CONVOLUTION_STRIDE_FORWARD(input,output.conv1_1,weight.weight1_1,weight.bias1_1,action);	\
	CONVOLUTION_PADDING_FORWARD(output.conv1_1, output.conv1_2,weight.weight1_2,weight.bias1_2,action);\
	CONVOLUTION_FORWARD_1D(input,output.conv_res,weight.conv1,action);									\
	OMP_PARA_INTERNAL																	\
	FOREACH(i, GETCOUNT(output.conv1_2))														\
		((double*)output.conv1_2)[i] += ((double*)output.conv_res)[i];									\
	CONVOLUTION_PADDING_FORWARD(output.conv1_2,output.conv2_1,weight.weight2_1,weight.bias2_1,action);\
	CONVOLUTION_PADDING_FORWARD(output.conv2_1, output.conv2_2,weight.weight2_2,weight.bias2_2,action);\
	OMP_PARA_INTERNAL																	\
	FOREACH(i, GETCOUNT(output.conv2_2))														\
		((double*)output.conv2_2)[i] += ((double*)output.conv1_2)[i];							\
}

#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	OMP_PARA_INTERNAL																	\
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

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	OMP_PARA_INTERNAL																	\
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

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	OMP_PARA_INTERNAL														\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
	OMP_PARA_INTERNAL														\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	OMP_PARA_INTERNAL															\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	OMP_PARA_INTERNAL															\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	OMP_PARA_INTERNAL															\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	OMP_PARA_INTERNAL															\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}

//((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y]; 这里是计算input中将要迭代至的上一层的值对Loss的梯度，以便上一层用
//wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];	这里是计算当前梯度更新层，要更新的梯度
double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

static void forward(ResNet18 *resnet, Feature *features, double(*action)(double))
{
	CONVOLUTION_PADDING_FORWARD(features->input,features->conv1,resnet->weight1,resnet->bias1,action);
	SUBSAMP_MAX_FORWARD(features->conv1, features->pool1);
	ResBlock1_Forward(features->pool1,features->res1f,resnet->res_block1,action);
	ResBlock2_Forward(features->res1f.conv2_2,features->res2f,resnet->res_block2,action);
	ResBlock2_Forward(features->res2f.conv2_2,features->res3f,resnet->res_block3,action);
	ResBlock2_Forward(features->res3f.conv2_2,features->res4f,resnet->res_block4,action);
	SUBSAMP_MAX_FORWARD(features->res4f.conv2_2, features->pool2);
	DOT_PRODUCT_FORWARD(features->pool2, features->output, resnet->fc, resnet->bias_fc, action);

}

static void backward(ResNet18 *resnet, ResNet18 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{

	DOT_PRODUCT_BACKWARD(features->pool2, errors->pool2, errors->output, resnet->fc, deltas->fc, deltas->bias_fc, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->res4f.conv2_2, errors->res4f.conv2_2, errors->pool2);
	ResBlock2_Backward_FromPool(features->res4f, errors->res4f, resnet->res_block4, deltas->res_block4,actiongrad,features->res3f.conv2_2,errors->res3f.conv2_2);
	ResBlock2_Backward_FromPool(features->res3f, errors->res3f, resnet->res_block3, deltas->res_block3,actiongrad,features->res2f.conv2_2,errors->res2f.conv2_2);
	ResBlock2_Backward_FromPool(features->res2f, errors->res2f, resnet->res_block2, deltas->res_block2,actiongrad,features->res1f.conv2_2,errors->res1f.conv2_2);
	ResBlock1_Backward(features->res1f,errors->res1f,resnet->res_block1,deltas->res_block1,actiongrad,features->pool1,errors->pool1);
	SUBSAMP_MAX_BACKWARD(features->conv1, errors->conv1, errors->pool1);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->conv1, resnet->weight1, deltas->weight1, deltas->bias1, actiongrad);
	// SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	// CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, resnet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	// SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	// CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, resnet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}


static inline void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = 32*32;
	double mean[3]={0}, std[3] = {0};
	FOREACH(i,3)
	{
		FOREACH(j, 32)
			FOREACH(k, 32)
		{
			mean[i] += input[i][j][k];
			std[i] += input[i][j][k] * input[i][j][k];
		}
		mean[i] /= sz;
		std[i] = sqrt(std[i] / sz - mean[i]*mean[i]);
	}

	FOREACH(i,3)
	FOREACH(j, 32)
		FOREACH(k, 32)
	{
		layer0[i][j][k] = (input[i][j][k] - mean[i]) / std[i];
	}
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
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;//This is the original implementation, not sure if it is the best, you can replace with better optimization function
		//loss[i] *= (i == label) - inner;//Here is the actual implementation on mathematics
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
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
			result = i;
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

ResNet18 deltas={0};
void TrainBatch(ResNet18 *resnet, image *inputs, uint8 *labels, int batchSize)
{

	double buffer[GETCOUNT(ResNet18)] = { 0 };
	int i = 0;
	ofstream myfile("res18_parallel_cifar10_batch1_100_training.txt",ios::out|ios::app);
	start_timer_res18();
#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		// ResNet18 *deltas = (ResNet18 *)malloc(sizeof(ResNet18));
		load_input(&features, inputs[i]);
		forward(resnet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(resnet, &deltas, &errors, &features, relugrad);
		#pragma omp critical
		{
			FOREACH(j, GETCOUNT(ResNet18))
				buffer[j] += ((double *)&deltas)[j];
		}

	}

	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(ResNet18))
		((double *)resnet)[i] += k * buffer[i];
	stop_timer_res18();	
	std::cout << "The time of a batch of 1 image trainng of Res18 single is " << std::chrono::duration_cast<std::chrono::milliseconds>(duration_res18).count() << std::endl;
	myfile << ((float)std::chrono::duration_cast<std::chrono::milliseconds>(duration_res18).count())/1000;
	myfile << '\n';
	myfile.close();

}

void Train(ResNet18 *resnet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	ResNet18 deltas = { 0 };
	load_input(&features, input);
	forward(resnet, &features, relu);
	load_target(&features, &errors, label);
	backward(resnet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(ResNet18))
		((double *)resnet)[i] += ALPHA * ((double *)&deltas)[i];
}

uint8 Predict(ResNet18 *resnet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(resnet, &features, relu);
	return get_result(&features, count);
}

void Initial(ResNet18 *resnet)
{
	for (double *pos = (double *)resnet->weight1; pos < (double *)resnet->bias1; *pos++ = f64rand());
	// for (double *pos = (double *)resnet->weight0_1; pos < (double *)resnet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	// for (double *pos = (double *)resnet->weight2_3; pos < (double *)resnet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	// for (double *pos = (double *)resnet->weight4_5; pos < (double *)resnet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	// for (double *pos = (double *)resnet->weight5_6; pos < (double *)resnet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	// for (int *pos = (int *)resnet->bias0_1; pos < (int *)(resnet + 1); *pos++ = 0);
}
