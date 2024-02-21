#include "vgg.h"
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
#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))
#define OMP_PARA_INTERNAL _Pragma("omp parallel for")
#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)
decltype(std::chrono::system_clock::now()) start_vgg;
decltype(std::chrono::system_clock::now() - std::chrono::system_clock::now()) duration_vgg;
void start_timer_vgg() {
	start_vgg = std::chrono::system_clock::now();
}
void stop_timer_vgg() {
	duration_vgg = std::chrono::system_clock::now() - start_vgg;
}
#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

// #define CONVOLUTE_Backward_Weight_VALID(input,output,weight)							\
// {																						\
// 	FOREACH(o0,GETLENGTH(output))														\
// 		FOREACH(o1,GETLENGTH(*(output)))												\
// 			FOREACH(w0,GETLENGTH(weight)-2*PADDING)												\
// 				FOREACH(w1,GETLENGTH(*(weight)-2*PADDING))										\
// 					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0+PADDING][w1+PADDING];\
// }

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}
#define ToPad(input_pad,input,PADDING){													\
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
#define CONVOLUTION_FORWARD(input,output,weight,bias,action)						\
{																					\
	double input_pad[GETLENGTH(input)][GETLENGTH(*input)+2*PADDING][GETLENGTH(**input)+2*PADDING]={0};	\
	ToPad(input_pad,input,PADDING);													\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input_pad[x], output[y], weight[x][y]);				\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
	extractPad(input,input_pad,PADDING);										\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	double input_pad[GETLENGTH(input)][GETLENGTH(*input)+2*PADDING][GETLENGTH(**input)+2*PADDING]={0};	\
	double inerror_pad[GETLENGTH(input)][GETLENGTH(*input)+2*PADDING][GETLENGTH(**input)+2*PADDING]={0};\
	ToPad(input_pad,input,PADDING);													\
	ToPad(inerror_pad,inerror,PADDING);												\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror_pad[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input_pad)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input_pad[x], wd[x][y], outerror[y]);\
	extractPad(input,input_pad,PADDING);										\
	extractPad(inerror,inerror_pad,PADDING);										\
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

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}

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

static void forward(VGG *vggnet, Feature *features, double(*action)(double))
{
	CONVOLUTION_FORWARD(features->input, features->layer1_conv1, vggnet->weight1_1, vggnet->bias1_1, action);
	CONVOLUTION_FORWARD(features->layer1_conv1, features->layer1_conv2, vggnet->weight1_2, vggnet->bias1_2, action);
	SUBSAMP_MAX_FORWARD(features->layer1_conv2, features->layer1_pool);
	
	CONVOLUTION_FORWARD(features->layer1_pool, features->layer2_conv1, vggnet->weight2_1, vggnet->bias2_1, action);
	CONVOLUTION_FORWARD(features->layer2_conv1, features->layer2_conv2, vggnet->weight2_2, vggnet->bias2_2, action);
	SUBSAMP_MAX_FORWARD(features->layer2_conv2, features->layer2_pool);

	CONVOLUTION_FORWARD(features->layer2_pool, features->layer3_conv1, vggnet->weight3_1, vggnet->bias3_1, action);
	CONVOLUTION_FORWARD(features->layer3_conv1, features->layer3_conv2, vggnet->weight3_2, vggnet->bias3_2, action);
	CONVOLUTION_FORWARD(features->layer3_conv2, features->layer3_conv3, vggnet->weight3_3, vggnet->bias3_3, action);
	CONVOLUTION_FORWARD(features->layer3_conv3, features->layer3_conv4, vggnet->weight3_4, vggnet->bias3_4, action);
	SUBSAMP_MAX_FORWARD(features->layer3_conv4, features->layer3_pool);

	CONVOLUTION_FORWARD(features->layer3_pool, features->layer4_conv1, vggnet->weight4_1, vggnet->bias4_1, action);
	CONVOLUTION_FORWARD(features->layer4_conv1, features->layer4_conv2, vggnet->weight4_2, vggnet->bias4_2, action);
	CONVOLUTION_FORWARD(features->layer4_conv2, features->layer4_conv3, vggnet->weight4_3, vggnet->bias4_3, action);
	CONVOLUTION_FORWARD(features->layer4_conv3, features->layer4_conv4, vggnet->weight4_4, vggnet->bias4_4, action);
	SUBSAMP_MAX_FORWARD(features->layer4_conv4, features->layer4_pool);

	CONVOLUTION_FORWARD(features->layer4_pool, features->layer5_conv1, vggnet->weight5_1, vggnet->bias5_1, action);
	CONVOLUTION_FORWARD(features->layer5_conv1, features->layer5_conv2, vggnet->weight5_2, vggnet->bias5_2, action);
	CONVOLUTION_FORWARD(features->layer5_conv2, features->layer5_conv3, vggnet->weight5_3, vggnet->bias5_3, action);
	CONVOLUTION_FORWARD(features->layer5_conv3, features->layer5_conv4, vggnet->weight5_4, vggnet->bias5_4, action);
	SUBSAMP_MAX_FORWARD(features->layer5_conv4, features->layer5_pool);

	DOT_PRODUCT_FORWARD(features->layer5_pool, features->fc1, vggnet->fc1, vggnet->bias_fc1, action);
	DOT_PRODUCT_FORWARD(features->fc1, features->fc2, vggnet->fc2, vggnet->bias_fc2, action);
	DOT_PRODUCT_FORWARD(features->fc2, features->output, vggnet->fc3, vggnet->bias_fc3, action);

}

static void backward(VGG *vggnet, VGG *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->fc2, errors->fc2, errors->output, vggnet->fc3, deltas->fc3, deltas->bias_fc3, actiongrad);
	DOT_PRODUCT_BACKWARD(features->fc1, errors->fc1, errors->fc2, vggnet->fc2, deltas->fc2, deltas->bias_fc2, actiongrad);
	DOT_PRODUCT_BACKWARD(features->layer5_pool, errors->layer5_pool, errors->fc1, vggnet->fc1, deltas->fc1, deltas->bias_fc1, actiongrad);

	SUBSAMP_MAX_BACKWARD(features->layer5_conv4, errors->layer5_conv4, errors->layer5_pool);
	CONVOLUTION_BACKWARD(features->layer5_conv3, errors->layer5_conv3, errors->layer5_conv4, vggnet->weight5_4, deltas->weight5_4, deltas->bias5_4, actiongrad);
	CONVOLUTION_BACKWARD(features->layer5_conv2, errors->layer5_conv2, errors->layer5_conv3, vggnet->weight5_3, deltas->weight5_3, deltas->bias5_3, actiongrad);
	CONVOLUTION_BACKWARD(features->layer5_conv1, errors->layer5_conv1, errors->layer5_conv2, vggnet->weight5_2, deltas->weight5_2, deltas->bias5_2, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4_pool, errors->layer4_pool, errors->layer5_conv1, vggnet->weight5_1, deltas->weight5_1, deltas->bias5_1, actiongrad);


	SUBSAMP_MAX_BACKWARD(features->layer4_conv4, errors->layer4_conv4, errors->layer4_pool);
	CONVOLUTION_BACKWARD(features->layer4_conv3, errors->layer4_conv3, errors->layer4_conv4, vggnet->weight4_4, deltas->weight4_4, deltas->bias4_4, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4_conv2, errors->layer4_conv2, errors->layer4_conv3, vggnet->weight4_3, deltas->weight4_3, deltas->bias4_3, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4_conv1, errors->layer4_conv1, errors->layer4_conv2, vggnet->weight4_2, deltas->weight4_2, deltas->bias4_2, actiongrad);
	CONVOLUTION_BACKWARD(features->layer3_pool, errors->layer3_pool, errors->layer4_conv1, vggnet->weight4_1, deltas->weight4_1, deltas->bias4_1, actiongrad);

	SUBSAMP_MAX_BACKWARD(features->layer3_conv4, errors->layer3_conv4, errors->layer3_pool);
	CONVOLUTION_BACKWARD(features->layer3_conv3, errors->layer3_conv3, errors->layer3_conv4, vggnet->weight3_4, deltas->weight3_4, deltas->bias3_4, actiongrad);
	CONVOLUTION_BACKWARD(features->layer3_conv2, errors->layer3_conv2, errors->layer3_conv3, vggnet->weight3_3, deltas->weight3_3, deltas->bias3_3, actiongrad);
	CONVOLUTION_BACKWARD(features->layer3_conv1, errors->layer3_conv1, errors->layer3_conv2, vggnet->weight3_2, deltas->weight3_2, deltas->bias3_2, actiongrad);
	CONVOLUTION_BACKWARD(features->layer2_pool, errors->layer2_pool, errors->layer3_conv1, vggnet->weight3_1, deltas->weight3_1, deltas->bias3_1, actiongrad);


	SUBSAMP_MAX_BACKWARD(features->layer2_conv2, errors->layer2_conv2, errors->layer2_pool);
	CONVOLUTION_BACKWARD(features->layer2_conv1, errors->layer2_conv1, errors->layer2_conv2, vggnet->weight2_2, deltas->weight2_2, deltas->bias2_2, actiongrad);
	CONVOLUTION_BACKWARD(features->layer1_pool, errors->layer1_pool, errors->layer2_conv1, vggnet->weight2_1, deltas->weight2_1, deltas->bias2_1, actiongrad);


	SUBSAMP_MAX_BACKWARD(features->layer1_conv2, errors->layer1_conv2, errors->layer1_pool);
	CONVOLUTION_BACKWARD(features->layer1_conv1, errors->layer1_conv1, errors->layer1_conv2, vggnet->weight1_2, deltas->weight1_2, deltas->bias1_2, actiongrad);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1_conv1, vggnet->weight1_1, deltas->weight1_1, deltas->bias1_1, actiongrad);


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
		loss[i] *= (i == label) - loss[i] - inner;//这是原来的实现，不确定是不是更好，或者现代用的优化函数
		//loss[i] *= (i == label) - inner;//这是实际数学计算中的实现
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


void TrainBatch(VGG *vggnet, image *inputs, uint8 *labels, int batchSize)
{

	double buffer[GETCOUNT(VGG)] = { 0 };
	int i = 0;
	int right=0;

	//omp_set_num_threads(2);
// #pragma omp parallel for
// for (int i = 0; i < 50; i++) {
// 	for (int j = 0; j < 4; j++) {
// 		printf("#######%d %d\n",i,j);
// 	}
// }
   	ofstream   myfile("vgg_single_cifar_batch300_900_training.txt",ios::out|ios::app);
	start_timer_vgg();
// #pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		// printf("Here is TrainBatch %d Thread:%d\n",i,omp_get_thread_num());
		// printf("The total thread number is %d\n",omp_get_num_threads());
		Feature features = { 0 };
		Feature errors = { 0 };
		VGG	deltas = { 0 };
		load_input(&features, inputs[i]);
		// printf("I am thread %d for %d\n",omp_get_thread_num(),i);

		// FOREACH(k,GETLENGTH(features.input))
		// 	FOREACH(j,GETLENGTH(*features.input))
		// 		FOREACH(z,GETLENGTH(**features.input))
		// 			printf("%f", (features.input[k][j][z]));

		forward(vggnet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(vggnet, &deltas, &errors, &features, relugrad);
		// #pragma omp critical
		// {
			FOREACH(j, GETCOUNT(VGG))
				buffer[j] += ((double *)&deltas)[j];
		// }
	// right+=(get_result(&features,10) == labels[i]);
	// printf("right:%d Number:%d, Thread:%d \n", right,i,omp_get_thread_num());
	}

	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(VGG))
		((double *)vggnet)[i] += k * buffer[i];
	stop_timer_vgg();
	std::cout << "The time of each batch=100 image trainng of VGG  single is " << std::chrono::duration_cast<std::chrono::milliseconds>(duration_vgg).count() << std::endl;
	myfile << ((float)std::chrono::duration_cast<std::chrono::milliseconds>(duration_vgg).count())/1000;
	myfile << '\n';
	myfile.close();
}

void Train(VGG *vggnet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	VGG deltas = { 0 };
	load_input(&features, input);
	forward(vggnet, &features, relu);
	load_target(&features, &errors, label);
	backward(vggnet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(VGG))
		((double *)vggnet)[i] += ALPHA * ((double *)&deltas)[i];
}

uint8 Predict(VGG *vggnet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(vggnet, &features, relu);
	return get_result(&features, count);
}

void Initial(VGG *vggnet)
{
	for (double *pos = (double *)vggnet->weight1_1; pos < (double *)vggnet->bias1_1; *pos++ = f64rand());
	for (double *pos = (double *)vggnet->weight1_1; pos < (double *)vggnet->weight1_2; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)vggnet->weight1_2; pos < (double *)vggnet->weight2_1; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER1 + LAYER1))));
	for (double *pos = (double *)vggnet->weight2_1; pos < (double *)vggnet->weight2_2; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER1 + LAYER2))));
	for (double *pos = (double *)vggnet->weight3_1; pos < (double *)vggnet->weight3_2; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER2))));
	for (double *pos = (double *)vggnet->weight3_2; pos < (double *)vggnet->weight3_3; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)vggnet->weight3_3; pos < (double *)vggnet->weight3_4; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER3 + LAYER3))));
	for (double *pos = (double *)vggnet->weight3_4; pos < (double *)vggnet->weight4_1; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER3 + LAYER3))));
	for (double *pos = (double *)vggnet->weight4_1; pos < (double *)vggnet->weight4_2; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER3 + LAYER3))));
	for (double *pos = (double *)vggnet->weight4_2; pos < (double *)vggnet->weight4_3; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER3 + LAYER4))));
	for (double *pos = (double *)vggnet->weight4_3; pos < (double *)vggnet->weight4_4; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER4))));
	for (double *pos = (double *)vggnet->weight4_4; pos < (double *)vggnet->weight5_1; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER4))));
	for (double *pos = (double *)vggnet->weight5_1; pos < (double *)vggnet->weight5_2; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (double *pos = (double *)vggnet->weight5_2; pos < (double *)vggnet->weight5_3; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER5 + LAYER5))));
	for (double *pos = (double *)vggnet->weight5_3; pos < (double *)vggnet->weight5_4; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER5 + LAYER5))));
	for (double *pos = (double *)vggnet->weight5_4; pos < (double *)vggnet->fc1; *pos++ *= sqrt(20.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER5 + LAYER5))));
	for (double *pos = (double *)vggnet->fc1; pos < (double *)vggnet->fc2; *pos++ *= sqrt(20.0 / (LAYER5*LENGTH_FEATURE5*LENGTH_FEATURE5*FC1)));
	for (double *pos = (double *)vggnet->fc2; pos < (double *)vggnet->fc3; *pos++ *= sqrt(20.0 / (FC1*FC2)));
	for (double *pos = (double *)vggnet->fc3; pos < (double *)vggnet->bias1_1; *pos++ *= sqrt(20.0 / (FC2*OUTPUT)));
	for (int *pos = (int *)vggnet->bias1_1; pos < (int *)(vggnet + 1); *pos++ = 0);
}
