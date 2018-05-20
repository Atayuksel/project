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
void bpoutput(double* outputUnitValues, double* hiddenUnitValues, \
								int lexiconSize, double* updateWeightArr, \
								int targetWordIdx) {

	int i = threadIdx.x;

	double *tmp;

	cublasHandle_t cubpout;
	cublasCreate(&cubpout);

	cudaError_t errSync = cudaMalloc(&tmp, (1 * lexiconSize * sizeof(double)));
	printf("tmp allocation in thread %d: %s\n",i, cudaGetErrorString(errSync));

	memcpy(tmp, outputUnitValues, (lexiconSize * sizeof(double)));

	double inv = -1;
	cublasDscal(cubpout, lexiconSize, &inv, tmp, 1);

	tmp[targetWordIdx] = tmp[targetWordIdx] + 1;
	double mult = 0.1 * hiddenUnitValues[i];

	cublasDscal(cubpout, lexiconSize, &mult, tmp, 1);

	double *IdxUpWeight = updateWeightArr + i * lexiconSize;

	memcpy(IdxUpWeight, tmp, (lexiconSize * sizeof(double)));
	printf("Exiting threadID: %d\n", i);
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

	printf("Size dataset is %d\n", sizeDataset);

	for (int idx = 0; idx < 1; idx++) {
		int sourceWordIdx = source_idx[idx];
		int targetWordIdx = target_idx[idx];

		// obtain values of hidden units to varHiddenUnits
		double *hiddenWeightsStartPointer = hiddenWeights + (sourceWordIdx * hiddenUnitCount);
		memcpy(varHiddenUnits, hiddenWeightsStartPointer, (hiddenUnitCount * sizeof(double)));

		printf("[ ");
		for (int i = 0; i < hiddenUnitCount; i++) {
			printf("%f, ", varHiddenUnits[i]);
		}
		printf(" ]\n");

		// obtain values of output units via dgemm
		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasDgemm(handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			lexiconSize, 1, hiddenUnitCount,
			alpha,
			out, lexiconSize,
			varHiddenUnits, hiddenUnitCount,
			beta,
			varOutputUnits, lexiconSize);

		cudaDeviceSynchronize();

		cudaError_t errSync = cudaGetLastError();
		printf("After dgemm function: %s\n", cudaGetErrorString(errSync));

		/*printf("[ ");
		for (int i = 0; i < lexiconSize; i++) {
			printf("%f, ", varOutputUnits[i]);
		}
		printf(" ]\n");*/

		double sumOutputUnits = 0.0;
		for (int i = 0; i < lexiconSize; i++) {
			sumOutputUnits = sumOutputUnits + varOutputUnits[i];
		}

		double valScal = (double)1 / sumOutputUnits;
		cublasDscal(handle, lexiconSize, &valScal, varOutputUnits, 1);

		errSync = cudaGetLastError();
		printf("After dscal func: %s\n", cudaGetErrorString(errSync));

		// calculation update values for output layer weights
		double *upOutWeights, *IdxUpWeight;

		errSync = cudaMalloc(&upOutWeights, (hiddenUnitCount * lexiconSize * sizeof(double)));
		printf("after cudamalloc: %s\n", cudaGetErrorString(errSync));

		double inv = -1;
		double mult;
		double zero = 0;

		bpoutput<<<1,100>>>(varOutputUnits, varHiddenUnits, lexiconSize, \
												upOutWeights, targetWordIdx);

		cudaDeviceSynchronize();

		// printf("[ ");
		// for (int i = 0; i < lexiconSize; i++) {
		// 	printf("%f, ", tmp[i]);
		// }
		// printf(" ]\n");

		// calculation update values for hidden layer weights
		double *UpHidWeights, *tmp, *tmpS, *IdxPointer;
		cudaMalloc(&UpHidWeights, (hiddenUnitCount * lexiconSize * sizeof(double)));
		cudaMalloc(&tmpS, (lexiconSize * sizeof(double)));
		cudaMalloc(&tmp, (lexiconSize * sizeof(double)));
		for (int i = 0; i < hiddenUnitCount; i++) {
			IdxPointer = i * lexiconSize + out;
			memcpy(tmpS, IdxPointer, (lexiconSize * sizeof(double)));
			memcpy(tmp, varOutputUnits, (lexiconSize * sizeof(double)));
			cublasDscal(handle, lexiconSize, &inv, tmp, 1);
			tmp[targetWordIdx] = tmp[targetWordIdx] + 1;
			for (int j = 0; j < lexiconSize; j++) {
				tmp[j] = tmp[j] * tmpS[j];
			}
			cublasDscal(handle, lexiconSize, &zero, tmpS, 1);
			tmpS[sourceWordIdx] = tmp[sourceWordIdx] * learningFactor;
			IdxPointer = i * lexiconSize + UpHidWeights;
			memcpy(IdxPointer, tmpS, (lexiconSize * sizeof(double)));
		}

		errSync = cudaGetLastError();
		printf("After calculation update values for output weights: %s\n", cudaGetErrorString(errSync));

		int hiddenWeightIdx;
		int hiddenUnitIdx = 0;
		int lexiconIdx = 0;
		for (int i = 0; i < lexiconSize * hiddenUnitCount; i++) {
			hiddenUnitIdx = i / lexiconSize;
			lexiconIdx = i - (lexiconSize * hiddenUnitIdx);
			hiddenWeightIdx = (hiddenUnitCount * lexiconIdx) + hiddenUnitIdx;
			hiddenWeights[hiddenWeightIdx] = hiddenWeights[hiddenWeightIdx] + UpHidWeights[i];
			out[i] = out[i] + upOutWeights[i];
		}
		errSync = cudaGetLastError();
		printf("After calculating hidden weights: %s\n", cudaGetErrorString(errSync));
		// printf("%d", idx);

	}
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
	double a = -0.01, b = 0.01;
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


	// print weights before kernel
	/*printf("[ ");
	for (int i = 0; i < N; i++) {
		printf("%f, ", randomNumsa[i]);
	}
	printf(" ]\n");
	printf("[ ");
	for (int i = 0; i < N; i++) {
		printf("%f, ", randomNumsb[i]);
	}
	printf(" ]\n");*/

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
	errSync = cudaDeviceSetLimit(limit, total*10);
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

	/*printf("[ ");
	for (int i = 0; i < N; i++) {
		printf("%f, ", randomNumsa[i]);
	}
	printf(" ]\n");

	printf("[ ");
	for (int i = 0; i < N; i++) {
		printf("%f, ", randomNumsb[i]);
	}
	printf(" ]\n");*/

	cudaFree(cudaRandA);
	cudaFree(cudaRandB);

	errSync = cudaGetLastError();
	printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Time consumed for a single iter: %f\n", time_spent);
	return Py_None;
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
