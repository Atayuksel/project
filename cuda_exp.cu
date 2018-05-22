#include <Python.h>
#include <mkl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "stdio.h"
#include <numpy/arrayobject.h>
#include "cublas_v2.h"

#define SEED    1
#define BRNG    VSL_BRNG_MCG31
#define METHOD  0

__global__
void parallelCpyArr(double *a, double *b, int length){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i = index; i < length; i += stride){
		a[i] = b[i];
	}
}

__global__
void divideArr(double *arr, int length, double total){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		double a = 0;
		for (int i = index; i < length; i += stride){
			a = arr[i];
			arr[i] = arr[i] / total;
			// if (isnan(arr[i]) && !isnan(a)){
			// 	printf("operation: %f = %f / %f\n", arr[i], a, total);
			// }
		}
}

__global__
void applyExponArr(double *outputValues, int lengthArr){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < lengthArr; i += stride){
		outputValues[i] = exp(outputValues[i]);
	}
}

__global__
void updateWeights(double *outputWeights, double *outputWeightsUpdateTable, \
										double *hiddenWeights, double *hiddenWeightsUpdateTable, \
										int lexiconSize, int hiddenUnitCount){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
	int hiddenWeightIdx;
	int hiddenUnitIdx = 0;
	int lexiconIdx = 0;
	for(int i = index; i < (hiddenUnitCount * lexiconSize); i += stride){
		hiddenUnitIdx = i / lexiconSize;
		lexiconIdx = i - (lexiconSize * hiddenUnitIdx);
		hiddenWeightIdx = (hiddenUnitCount * lexiconIdx) + hiddenUnitIdx;
		hiddenWeights[hiddenWeightIdx] = hiddenWeights[hiddenWeightIdx] \
		+ hiddenWeightsUpdateTable[i];
		outputWeights[i] = outputWeights[i] + outputWeightsUpdateTable[i];
	}
}

__global__
void dotProductKernel(double *a, double *b, double *c, int length){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
	if (index < length){
		for (int i = index; i < length; i += stride){
			c[i] = a[i] * b[i];
		}
	}
}

__global__
void bphidden(double *outputWeights, double *outputUnitValues, \
 								int targetWordIdx, int sourceWordIdx, \
								double *hiddenWeightsUpdateTable, int lexiconSize, int hiddenUnitCount) {
	int index = threadIdx.x;
	int stride = blockDim.x;

	double inv = -1;
	double zero = 0;

	double *tmp2, *tmp;
	cudaMalloc(&tmp2, (1 * lexiconSize * sizeof(double)));
	cudaMalloc(&tmp, (1 * lexiconSize * sizeof(double)));
	cublasHandle_t cubphid;
	cublasCreate(&cubphid);

	for (int i = index; i < hiddenUnitCount; i += stride){


		double *IdxPointer = i * lexiconSize + outputWeights;

		// memcpy(tmp2, IdxPointer, (lexiconSize * sizeof(double)));
		// memcpy(tmp, outputUnitValues, (lexiconSize * sizeof(double)));

		parallelCpyArr<<<1,2024>>>(tmp2, IdxPointer, lexiconSize);
		cudaDeviceSynchronize();

		parallelCpyArr<<<1,2024>>>(tmp, outputUnitValues, lexiconSize);
		cudaDeviceSynchronize();

		cublasDscal(cubphid, lexiconSize, &inv, tmp, 1);
		cudaDeviceSynchronize();
		tmp[targetWordIdx] = tmp[targetWordIdx] + 1;
		dotProductKernel<<<1,1024>>>(tmp, tmp2, tmp, lexiconSize);
		cudaDeviceSynchronize();
		cublasDscal(cubphid, lexiconSize, &zero, tmp2, 1);
		cudaDeviceSynchronize();
		tmp2[sourceWordIdx] = tmp[sourceWordIdx] * 0.1;
		IdxPointer = i * lexiconSize + hiddenWeightsUpdateTable;
		// memcpy(IdxPointer, tmp2, (lexiconSize * sizeof(double)));
		parallelCpyArr<<<1,2024>>>(IdxPointer, tmp2, lexiconSize);
		cudaDeviceSynchronize();

	}
	cublasDestroy(cubphid);
	cudaFree(tmp);
	cudaFree(tmp2);
}

__global__
void bpoutput(double* outputUnitValues, double* hiddenUnitValues, \
								int lexiconSize, double* updateWeightArr, \
								int targetWordIdx, int hiddenUnitCount) {

	int index = threadIdx.x;
	int stride = blockDim.x;

	double *tmp;
	double inv = -1;
	double mult;

	cublasHandle_t cubpout;
	cublasCreate(&cubpout);

	cudaMalloc(&tmp, (1 * lexiconSize * sizeof(double)));
	for(int i = index; i < hiddenUnitCount; i += stride){
		// memcpy(tmp, outputUnitValues, (lexiconSize * sizeof(double)));
		parallelCpyArr<<<1,1024>>>(tmp, outputUnitValues, lexiconSize);
		cudaDeviceSynchronize();
		cublasDscal(cubpout, lexiconSize, &inv, tmp, 1);
		cudaDeviceSynchronize();
		tmp[targetWordIdx] = tmp[targetWordIdx] + 1;
		mult = 0.1 * hiddenUnitValues[i];
		cublasDscal(cubpout, lexiconSize, &mult, tmp, 1);
		cudaDeviceSynchronize();
		double *IdxUpWeight = updateWeightArr + i * lexiconSize;
		// memcpy(IdxUpWeight, tmp, (lexiconSize * sizeof(double)));
		parallelCpyArr<<<1,1024>>>(IdxUpWeight, tmp, lexiconSize);
		cudaDeviceSynchronize();
	}
	cudaFree(tmp);
	cublasDestroy(cubpout);
}

__global__
static void _trainNetwork(int sizeDataset, int* source_idx, int*target_idx, int lexiconSize,
	int hiddenUnitCount, double* hiddenWeights, double* out){

	double *varHiddenUnits, *varOutputUnits;
	cudaMalloc(&varHiddenUnits, (hiddenUnitCount * sizeof(double)));
	cudaMalloc(&varOutputUnits, (lexiconSize * sizeof(double)));
	double learningFactor = 0.1;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	// bool beforeHDgemm = false;
	// bool afterHDgemm = false;
	//
	// bool beforefDgemm = false;
	// bool afterfDgemm = false;
	//
	// bool beforeoDgemm = false;
	
	bool flag = false;

	double *upOutWeights;
	cudaMalloc(&upOutWeights, (hiddenUnitCount * lexiconSize * sizeof(double)));
	double *UpHidWeights;
	cudaMalloc(&UpHidWeights, (hiddenUnitCount * lexiconSize * sizeof(double)));
	double *hiddenWeightsStartPointer;

	double total = 0;

	int sourceWordIdx, targetWordIdx;
	cublasHandle_t handle;
	cublasCreate(&handle);

	for (int idx = 0; idx < sizeDataset; idx += 1) {
		sourceWordIdx = source_idx[idx];
		targetWordIdx = target_idx[idx];

		hiddenWeightsStartPointer = hiddenWeights + (sourceWordIdx * hiddenUnitCount);
		// memcpy(varHiddenUnits, hiddenWeightsStartPointer, (hiddenUnitCount * sizeof(double)));
		cudaDeviceSynchronize();
		parallelCpyArr<<<1,1024>>>(varHiddenUnits, hiddenWeightsStartPointer, hiddenUnitCount);

		// beforeHDgemm = false;
		// afterHDgemm = false;
		// beforeoDgemm = false;
		// afteroDgemm = false;
		// beforefDgemm = false;
		// afterfDgemm = false;
		//
		// for(int i = 0; i < hiddenUnitCount; i++){
		// 	if(isnan(varHiddenUnits[i])) {
		// 		beforeHDgemm = true;
		// 	}
		// }
		//
		// for(int i = 0; i < lexiconSize * hiddenUnitCount; i++){
		// 	if(isnan(out[i])) {
		// 		beforeoDgemm = true;
		// 	}
		// }

		cudaDeviceSynchronize();

		cublasDgemm(handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			lexiconSize, 1, hiddenUnitCount,
			alpha,
			out, lexiconSize,
			varHiddenUnits, hiddenUnitCount,
			beta,
			varOutputUnits, lexiconSize);

		cudaDeviceSynchronize();

		applyExponArr<<<1,1024>>>(varOutputUnits, lexiconSize);
		cudaDeviceSynchronize();

		total = 0;
		for (int i = 0; i < lexiconSize; i++){
			total += varOutputUnits[i];
		}


		// flag = false;
		// for(int i = 0; i < lexiconSize; i++){
		// 	if(isnan(varOutputUnits[i])){
		// 		flag = true;
		// 	}
		// }
		divideArr<<<1,1024>>>(varOutputUnits, lexiconSize, total);
		cudaDeviceSynchronize();
		// for(int i = 0; i < lexiconSize; i++){
		// 	if(isnan(varOutputUnits[i]) && !flag){
		// 		printf("tada");
		// 	}
		// }

		// flag = false;
		// for(int i = 0; i < lexiconSize * hiddenUnitCount; i++){
		// 	if(isnan(upOutWeights[i])){
		// 		flag = true;
		// 	}
		// }

		bpoutput<<<1,100>>>(varOutputUnits, varHiddenUnits, lexiconSize, \
												upOutWeights, targetWordIdx, hiddenUnitCount);
		cudaDeviceSynchronize();

		// for(int i = 0; i < lexiconSize * hiddenUnitCount; i++){
		// 	if(isnan(upOutWeights[i]) && !flag){
		// 		printf("tada");
		// 	}
		// }



		bphidden<<<1,100>>>(out, varOutputUnits, targetWordIdx, sourceWordIdx, \
											UpHidWeights, lexiconSize, hiddenUnitCount);
		cudaDeviceSynchronize();
		// printf("1st layer backpropagation is finished\n");
		// printf("Calling update weights function\n");

		updateWeights<<<1,1024>>>(out, upOutWeights, \
															hiddenWeights, UpHidWeights, \
															lexiconSize, hiddenUnitCount);
		cudaDeviceSynchronize();
		// printf("Updating weights is finished\n");
		double progress = (double)((double)idx/(double)sizeDataset)*(double)100;
		printf("\r In progress %.2f", progress);
	}
	cublasDestroy(handle);

	// printf("\n");
}

static PyObject* trainNetwork(PyObject* self, PyObject* args)
{
	PyArrayObject *arr_np_source, *arr_np_target;
	int lexiconSize, numberHiddenUnit;
	int *cpu_arr_source, *cpu_arr_target;
	if (!PyArg_ParseTuple(args, "OOii", &arr_np_source, &arr_np_target, &lexiconSize, &numberHiddenUnit)) {
		return NULL;
	}

	int sizeTrainingSet = arr_np_source->dimensions[0];
	printf("Size of the dataset is %d \n", sizeTrainingSet);

	cpu_arr_source = (int*)malloc(sizeTrainingSet * sizeof(int));
	cpu_arr_target = (int*)malloc(sizeTrainingSet * sizeof(int));

	for (int i = 0; i < sizeTrainingSet; i++) {
		int *item_source;
		int *item_target;
		item_source = (int *)PyArray_GETPTR1(arr_np_source, i);
		item_target = (int *)PyArray_GETPTR1(arr_np_target, i);
		cpu_arr_source[i] = *item_source;
		cpu_arr_target[i] = *item_target;
	}

	// allocating and copying idx of word pairs to gpu memory
	int *gpu_arr_source, *gpu_arr_target;
	cudaMalloc(&gpu_arr_source, (sizeTrainingSet * sizeof(int)));
	cudaMalloc(&gpu_arr_target, (sizeTrainingSet * sizeof(int)));
	cudaMemcpy(gpu_arr_source, cpu_arr_source, (sizeTrainingSet * sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_arr_target, cpu_arr_target, (sizeTrainingSet * sizeof(int)), cudaMemcpyHostToDevice);

	printf("Lexicon size is %d \n", lexiconSize);
	printf("Number of hidden unit is %d \n", numberHiddenUnit);

	// generate random doubles between -0.01 and 0.01 with MKL vdRngUniform
	int N = numberHiddenUnit * lexiconSize;
	double *randomNumsa = (double*)malloc(N * sizeof(double));
	double *randomNumsb = (double*)malloc(N * sizeof(double));
	VSLStreamStatePtr stream;
	double a = -0.1, b = 0.1;
	vslNewStream(&stream, BRNG, SEED);
	vdRngUniform(METHOD, stream, N, randomNumsa, a, b);
	vdRngUniform(METHOD, stream, N, randomNumsb, a, b);
	vslDeleteStream(&stream);

	// allocate and copy random generated weights to gpu
	double *cudaRandA;
	double *cudaRandB;
	cudaMalloc(&cudaRandA, (N * sizeof(double)));
	cudaMalloc(&cudaRandB, (N * sizeof(double)));
	cudaError_t errSync = cudaGetLastError();
	printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	cudaMemcpy(cudaRandA, randomNumsa, (N * sizeof(double)), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaRandB, randomNumsb, (N * sizeof(double)), cudaMemcpyHostToDevice);
	errSync = cudaGetLastError();
	printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	errSync = cudaGetLastError();
	printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));

	size_t empty, total;
	cudaSetDevice(0);
	cuMemGetInfo(&empty, &total);
	printf("%d %d\n",empty/1024,total/1024);

	cudaLimit limit = cudaLimitMallocHeapSize ;
	errSync = cudaDeviceGetLimit(&total, limit);
	printf("cuda device get limit: %s\n", cudaGetErrorString(errSync));
	printf("Device Limit: %d\n", total/1024);
	errSync = cudaDeviceSetLimit(limit, total*50);
	printf("cuda set device limit %s\n", errSync);
	printf("new device limit is:%d\n", total*10/1024);

	clock_t begin = clock();
	_trainNetwork<<<1,1>>>(sizeTrainingSet, gpu_arr_source, gpu_arr_target, lexiconSize,
		numberHiddenUnit, cudaRandA, cudaRandB);

	cudaDeviceSynchronize();
	clock_t end = clock();
	errSync = cudaGetLastError();
	printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));

	cudaMemcpy(randomNumsa, cudaRandA, (N * sizeof(double)), cudaMemcpyDeviceToHost);
	cudaMemcpy(randomNumsb, cudaRandB, (N * sizeof(double)), cudaMemcpyDeviceToHost);

	errSync = cudaGetLastError();
	printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	errSync = cudaGetLastError();
	printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));

	cudaFree(cudaRandA);
	cudaFree(cudaRandB);

	errSync = cudaGetLastError();
	printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Time consumed for a single iter: %f\n", time_spent);

	npy_intp dims[1];
	dims[0] = numberHiddenUnit * lexiconSize;
	PyObject *result = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, randomNumsb);

	return result;
}

static PyMethodDef word2vecMethods[] = {
	{ "trainNetwork", trainNetwork, METH_VARARGS, "start training of network" },
{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef wordtovec = {
	PyModuleDef_HEAD_INIT,
	"word2vec",
	"word2vec_doc",
	-1,
	word2vecMethods
};

PyMODINIT_FUNC PyInit_wordtovec(void)
{
	PyObject *m;
	m = PyModule_Create(&wordtovec);
	import_array();
	if (m == NULL) {
		return NULL;
	}
	return m;
}
