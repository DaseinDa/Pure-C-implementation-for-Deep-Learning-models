#include "alexnet.h"
// #include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include <fstream>
#include <chrono>
#include <iostream>
using namespace std;
#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))
#define OMP_PARA_INTERNAL _Pragma("omp parallel for")
#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

decltype(std::chrono::system_clock::now()) start_alex;
decltype(std::chrono::system_clock::now() - std::chrono::system_clock::now()) duration_alex;
void start_timer_alexnet() {
	start_alex = std::chrono::system_clock::now();
}
void stop_timer_alexnet() {
	duration_alex = std::chrono::system_clock::now() - start_alex;
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

#define Matrix_FC(input,output,weight)											\	
{																	\	
	OMP_PARA_INTERNAL																	\	
	FOREACH(o0,GETLENGTH(input))			\
		FOREACH(o1, GETLENGTH(*input))				\
			FOREACH(o2, GETLENGTH(**input))								\	
				FOREACH(i0, GETLENGTH(output))	\
				(output[i0] += input[o0][o1][o2]* weight[o0*o1+o2][i0]);					\	
}																						\															


#define FC(input,output,weight)\
{\
	OMP_PARA_INTERNAL																	\
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
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
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
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
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
//len0是输入矩阵的行数长度除以输出矩阵的行数长度
//len1是输入矩阵的列数宽度除以输出矩阵的列数宽度
//len0和len1是池化核的大小
#define SUBSAMP_MAX_STRIDE_FORWARD(input,output,kernel_size,stride)								\
{																								\
	const int len0 = kernel_size;																\
	const int len1 = kernel_size;																\
	const int s = stride;																		\
	OMP_PARA_INTERNAL																	\
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


#define SUBSAMP_MAX_STRIDE_BACKWARD(input,inerror,outerror,kernel_size, stride)										\
{																								\
	const int len0 = kernel_size;																\
	const int len1 = kernel_size;																\
	const int s = stride;																		\
	OMP_PARA_INTERNAL																	\
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
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y]; 	\
	OMP_PARA_INTERNAL																	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}																			\
// ((double *)output)[j] = action(((double *)output)[j] + bias[j]);全连接计算后的激活函数
// ((double *)output)[y] += ((double *)input)[x] * weight[x][y];全连接计算
//input是三维的， output是一维的



// #define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
// {																				\
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
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	OMP_PARA_INTERNAL																	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	OMP_PARA_INTERNAL																	\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	OMP_PARA_INTERNAL																	\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}


#define PADDING_fill(input_pad, input)											\
{																				\
	OMP_PARA_INTERNAL																	\
	FOREACH(i,GETLENGTH(input))													\
		FOREACH(j,GETLENGTH(*input))											\
			FOREACH(k,GETLENGTH(*input))										\
				input_pad[i][j+PADDING][k+PADDING]=input[i][j][k];				\		
}																				\

#define PADDING_remove(input, input_pad)											\
{																				\
	OMP_PARA_INTERNAL																	\
	FOREACH(i,GETLENGTH(input))													\
		FOREACH(j,GETLENGTH(*input))											\
			FOREACH(k,GETLENGTH(*input))										\
				input[i][j][k]=input_pad[i][j+PADDING][k+PADDING];				\		
}																			

double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

static void forward(AlexNet *alexnet,Feature *features, Feature_Pad *features_pad, double(*action)(double))
{
	PADDING_fill(features_pad->input,features->input);
	CONVOLUTION_FORWARD(features_pad->input, features->layer1_conv, alexnet->weight1, alexnet->bias1, action);
	SUBSAMP_MAX_FORWARD(features->layer1_conv, features->layer1_pool);

	PADDING_fill(features_pad->layer1_pool,features->layer1_pool);
	CONVOLUTION_FORWARD(features_pad->layer1_pool, features->layer2_conv, alexnet->weight2, alexnet->bias2, action);
	SUBSAMP_MAX_FORWARD(features->layer2_conv, features->layer2_pool);

	PADDING_fill(features_pad->layer2_pool,features->layer2_pool);
	CONVOLUTION_FORWARD(features_pad->layer2_pool, features->layer3_conv, alexnet->weight3, alexnet->bias3, action);
	
	PADDING_fill(features_pad->layer3_conv,features->layer3_conv);
	CONVOLUTION_FORWARD(features_pad->layer3_conv, features->layer4_conv, alexnet->weight4, alexnet->bias4, action);

	PADDING_fill(features_pad->layer4_conv,features->layer4_conv);
	CONVOLUTION_FORWARD(features_pad->layer4_conv, features->layer5_conv, alexnet->weight5, alexnet->bias5, action);
	//SUBSAMP_MAX_FORWARD(features->layer5_conv, features->layer5_pool);
	SUBSAMP_MAX_STRIDE_FORWARD(features->layer5_conv, features->layer5_pool,MAXPOOL5_2_KERNEL,MAXPOOL5_2_STRIDE);

	// Matrix_FC(features->layer5_pool, features->fc1, alexnet->fc1);
	// FC(features->fc1, features->fc2, alexnet->fc2);
	// FC(features->fc2, features->fc3, alexnet->fc3);

	DOT_PRODUCT_FORWARD(features->layer5_pool, features->fc1, alexnet->fc1, alexnet->bias_fc1, action);
	DOT_PRODUCT_FORWARD(features->fc1, features->fc2, alexnet->fc2, alexnet->bias_fc2, action);
	DOT_PRODUCT_FORWARD(features->fc2, features->output, alexnet->fc3, alexnet->bias_fc3, action);

	// CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	// SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	// CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	// DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void backward(AlexNet *alexnet,AlexNet *deltas,Feature *errors, Feature *features, Feature_Pad *features_pad, Feature_Pad *errors_pad, double(*actiongrad)(double))
{
	//printf("Here is Backward\n");
	DOT_PRODUCT_BACKWARD(features->fc2, errors->fc2, errors->output, alexnet->fc3, deltas->fc3, deltas->bias_fc3, actiongrad);
	DOT_PRODUCT_BACKWARD(features->fc1, errors->fc1, errors->fc2, alexnet->fc2, deltas->fc2, deltas->bias_fc2, actiongrad);
	DOT_PRODUCT_BACKWARD(features->layer5_pool, errors->layer5_pool, errors->fc1,alexnet->fc1, deltas->fc1, deltas->bias_fc1, actiongrad)
	
	//SUBSAMP_MAX_BACKWARD(features->layer5_conv, errors->layer5_conv, errors->layer5_pool);
	SUBSAMP_MAX_STRIDE_BACKWARD(features->layer5_conv, errors->layer5_conv, errors->layer5_pool,MAXPOOL5_2_KERNEL,MAXPOOL5_2_STRIDE);

	CONVOLUTION_BACKWARD(features_pad->layer4_conv, errors_pad->layer4_conv, errors->layer5_conv, alexnet->weight5, deltas->weight5, deltas->bias5, actiongrad);
	PADDING_remove(errors->layer4_conv,errors_pad->layer4_conv);

	CONVOLUTION_BACKWARD(features_pad->layer3_conv, errors_pad->layer3_conv, errors->layer4_conv, alexnet->weight4, deltas->weight4, deltas->bias4, actiongrad);
	PADDING_remove(errors->layer3_conv,errors_pad->layer3_conv);

	CONVOLUTION_BACKWARD(features_pad->layer2_pool, errors_pad->layer2_conv, errors->layer3_conv, alexnet->weight3, deltas->weight3, deltas->bias3, actiongrad);
	PADDING_remove(errors->layer2_conv,errors_pad->layer2_conv);
	SUBSAMP_MAX_BACKWARD(features->layer2_conv, errors->layer2_conv, errors->layer2_pool);

	CONVOLUTION_BACKWARD(features_pad->layer1_pool, errors_pad->layer1_pool, errors->layer2_conv, alexnet->weight2, deltas->weight2, deltas->bias2, actiongrad);
	PADDING_remove(errors->layer1_pool,errors_pad->layer1_pool);
	SUBSAMP_MAX_BACKWARD(features->layer1_conv, errors->layer1_conv, errors->layer1_pool);

	CONVOLUTION_BACKWARD(features_pad->input, errors_pad->input, errors->layer1_conv, alexnet->weight1, deltas->weight1, deltas->bias1, actiongrad);
	PADDING_remove(errors->input,errors_pad->input);




	// CONVOLUTION_BACKWARD(features->layer4_conv, errors->layer4_conv, errors->layer3_conv, alexnet->weight4, deltas->weight4, deltas->bias4, actiongrad);

	// CONVOLUTION_BACKWARD(features->layer3_conv, errors->layer3_conv, errors->layer4_conv, alexnet->weight3, deltas->weight3, deltas->bias3, actiongrad);
	
	
	// SUBSAMP_MAX_BACKWARD(features->layer2_pool, errors->layer2_pool, errors->layer3_conv);
	// CONVOLUTION_BACKWARD(features->layer5_conv, errors->layer5_conv, errors->layer5_pool, alexnet->weight5, deltas->weight5, deltas->bias5, actiongrad);

	//CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	// SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	
	// CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
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
		loss[i] = 1. / res; //这里是softmax的输出值 
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];//inner 是正确结果在模型中的预测概率值
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;// 如果j=i,loss=(1-pi-pi)*pi, 如果j不等于i, loss=(-pj-pi)pj这是梯度
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));//计算softmax梯度的损失函数
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


void TrainBatch(AlexNet *alexnet, image *inputs, uint8 *labels, int batchSize)
{
	double buffer[GETCOUNT(AlexNet)] = { 0 };
	int i = 0;
	// int right=0;
	ofstream   myfile("alexnet_parallel_cifar_batch1000_6000_training.txt",ios::out|ios::app);
	start_timer_alexnet();

#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		// printf("thread:%d the %d sample in a batch\n",omp_get_thread_snum(),i);
		Feature features = { 0 };
		Feature errors = { 0 };
		AlexNet	deltas = { 0 };
		Feature_Pad features_pad = { 0 };
		Feature_Pad errors_pad = { 0 };
		load_input(&features, inputs[i]);
		forward(alexnet, &features, &features_pad, relu);
		load_target(&features, &errors, labels[i]);
		backward(alexnet, &deltas, &errors, &features,&features_pad,&errors_pad,relugrad);
		// #pragma omp critical
		{
			FOREACH(j, GETCOUNT(AlexNet))
				buffer[j] += ((double *)&deltas)[j];
		}
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(AlexNet))
		((double *)alexnet)[i] += k * buffer[i];
	// right+=(get_result(&features,10) == labels[i]);
	// printf("right:%d\n", right);
	stop_timer_alexnet();	
	std::cout << "The time of a batch of 1 image trainng of Alexnet single is " << std::chrono::duration_cast<std::chrono::milliseconds>(duration_alex).count() << std::endl;
	myfile << ((float)std::chrono::duration_cast<std::chrono::milliseconds>(duration_alex).count())/1000;
	myfile << '\n';
	myfile.close();

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

// uint8 Predict(AlexNet *alexnet, Feature *features, image input,uint8 count)
// {
// 	load_input(&features,input);
// 	forward(alexnet, relu);
// 	return get_result(&features, count);
// }
// static double sgx_rand(){
// 	double number;
// 	unsigned char rand_num[4] = "123";
// 	sgx_status_t rand_ret = sgx_read_rand(rand_num, sizeof(rand_num));
// 	    double d;
//     for (int i = 0; i < sizeof(rand_num); i++)
//     {
//         number = ((double*)&rand_num)[i];
//     }

// 	return number;

// }
void Initial(AlexNet *alexnet)
{
	for (double *pos = (double *)alexnet->weight1; pos < (double *)alexnet->bias1; *pos++ = f64rand());
	for (double *pos = (double *)alexnet->weight1; pos < (double *)alexnet->weight2; *pos++ *= sqrt(9.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)alexnet->weight2; pos < (double *)alexnet->weight3; *pos++ *= sqrt(9.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER1 + LAYER2))));
	for (double *pos = (double *)alexnet->weight3; pos < (double *)alexnet->weight4; *pos++ *= sqrt(9.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)alexnet->weight4; pos < (double *)alexnet->weight5; *pos++ *= sqrt(9.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER3 + LAYER4))));
	for (double *pos = (double *)alexnet->weight5; pos < (double *)alexnet->fc1; *pos++ *= sqrt(9.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));

	for (double *pos = (double *)alexnet->fc1; pos < (double *)alexnet->fc2; *pos++ *= sqrt(9.0 /(LAYER5 + FC1_OUTPUT) ));
	for (double *pos = (double *)alexnet->fc2; pos < (double *)alexnet->fc3; *pos++ *= sqrt(9.0 / (FC1_OUTPUT+FC2_OUTPUT)));
	for (double *pos = (double *)alexnet->fc3; pos < (double *)alexnet->bias1; *pos++ *= sqrt(9.0 / (FC2_OUTPUT+FC3_OUTPUT)));

	for (int *pos = (int *)alexnet->bias1; pos < (int *)(alexnet + 1); *pos++ = 0);
}
