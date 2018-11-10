#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Algorithm switch
#define UNIFORM_CROSSOVER 1
#define ISLAND_MIGRATION 1

// Constants
#define CHROMOSOME_LENGTH 30
#define GENERATIONS 100
#define GRID_DIMENSION 1
#define BLOCK_DIMENSION 256
#define POPULATION_SIZE BLOCK_DIMENSION*GRID_DIMENSION
#define MIGRATION_INTERVAL 20

// Probability and percents
#define SELECTION_FITTEST_PERCENT 20
#define MIGRATION_PERCENT 10
#define CROSSOVER_PROBABILITY 45
#define MUTATION_PROBABILITY 5

// Functions
#define CURAND(x) (int)floor(curand_uniform(&localState)*(x-1))

// Error checking in cuda functions
#define CUDA_ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	}
}


// Valid Genes
const char *GENES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890, .-;:_!\"#%&/\\()=?@${[]}";
const int GENE_LENGTH = 86;
__constant__ const char *GENES_D = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890, .-;:_!\"#%&/\\()=?@${[]}";
__constant__ const int GENE_LENGTH_D = 86;


// Structure for individual genome
typedef struct Genotype{
	char chromosome[CHROMOSOME_LENGTH];
	int fitness;
} Individual;


// Methods for critical section
__device__ volatile int sem = 0;

__device__ void acquireSemaphore(volatile int *lock){
	while (atomicCAS((int *)lock, 0, 1) != 0);
}

__device__ void releaseSemaphore(volatile int *lock){
	*lock = 0;
	__threadfence();
}


__global__ void geneticAlgorithmKernel(Individual *population, char *targetStr, int targetStringLength, curandState *state, int *convergeGen){
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i<POPULATION_SIZE){
		
		// Copy population from global to shared memory of each block
		__shared__ Individual sharedPop[BLOCK_DIMENSION];
		int j = threadIdx.x;

		if(j < BLOCK_DIMENSION){
			sharedPop[j] = population[i];
		}
		__syncthreads();
		
		// Getting state for random number generation
		curandState localState = state[i];
		
		int generation = 0;
		while(1){
			
			generation++;
			// Fitness calculation
			int tmpFitness = 0;
			for (int k=0; k<CHROMOSOME_LENGTH; k++){
				if(sharedPop[j].chromosome[k]!=targetStr[k]) tmpFitness++;
			}
			sharedPop[j].fitness = tmpFitness;
			if(tmpFitness == 0 && generation < convergeGen[0]) convergeGen[0] = generation;

			if(generation>=GENERATIONS) break;
	
			// Sort based on fitness
			for (int k=0; k<BLOCK_DIMENSION/2; k++){
				if(j%2==0 && j<BLOCK_DIMENSION-1){
					if(sharedPop[j+1].fitness < sharedPop[j].fitness){
						Individual temp = sharedPop[j];
						sharedPop[j] = sharedPop[j+1];
						sharedPop[j+1] = temp;
					}
				}
				__syncthreads();
				if(j%2!=0 && j<BLOCK_DIMENSION-1){
					if(sharedPop[j+1].fitness < sharedPop[j].fitness){
						Individual temp = sharedPop[j];
						sharedPop[j] = sharedPop[j+1];
						sharedPop[j+1] = temp;
					}
				}
				__syncthreads();
			}

			// Migration
			#if ISLAND_MIGRATION
			if(generation % MIGRATION_INTERVAL == 0){

				__syncthreads();
				if(j == 0)	acquireSemaphore(&sem);
				__syncthreads();

				int migrationPop = (int) ceil((float) blockDim.x * (float)MIGRATION_PERCENT/(float)100);
				if(j<migrationPop){
					population[i] = sharedPop[j];
				}
				else if(j<2*migrationPop) {
					int k = (blockDim.x * (blockIdx.x + 1) + threadIdx.x - migrationPop) % POPULATION_SIZE;
					sharedPop[j] = population[k];
				}

				__syncthreads();
				if(j == 0)	releaseSemaphore(&sem);
				__syncthreads();
			}
			#endif
	
			// Selection
			int fittest = (int)ceil((float) blockDim.x * (float)SELECTION_FITTEST_PERCENT/(float)100);
			if(j > fittest){
				int p1 = CURAND(fittest);
				int p2 = CURAND(fittest);
			
				// Crossover and mutation
				#if UNIFORM_CROSSOVER
				for(int k=0; k<CHROMOSOME_LENGTH; k++){		// Uniform crossover
                                        int prob = CURAND(100);
                                        if(prob<45)             sharedPop[j].chromosome[k] = sharedPop[p1].chromosome[k];
                                        else if(prob<90)        sharedPop[j].chromosome[k] = sharedPop[p2].chromosome[k];
                                        else                    sharedPop[j].chromosome[k] = GENES_D[CURAND(GENE_LENGTH_D)];
                                }
				#else
				int partition = CURAND(CHROMOSOME_LENGTH);	// Single partition crossover
                                for(int k=0; k<partition; k++){
                                        sharedPop[j].chromosome[k] = sharedPop[p1].chromosome[k];
                                        if(CURAND(100) < MUTATION_PROBABILITY){
                                                sharedPop[j].chromosome[k] = GENES_D[CURAND(GENE_LENGTH_D)];
                                        }
                                }
                                for(int k=partition; k<CHROMOSOME_LENGTH; k++){
                                        sharedPop[j].chromosome[k] = sharedPop[p2].chromosome[k];
                                        if(CURAND(100) < MUTATION_PROBABILITY){
                                                sharedPop[j].chromosome[k] = GENES_D[CURAND(GENE_LENGTH_D)];
                                        }
				}
				#endif
			}
		}

		// Copy results back to global memory
		if(j < BLOCK_DIMENSION){
			population[i] = sharedPop[j];
			//printf("%d ", i);
		}
	}
}


__global__ void setup_kernel(curandState *state){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, id, 0, &state[id]);
}


int main(int argc, char* argv[]){

	// Time measurement variables
	float h2dTime, d2hTime, kernelTime;

	// Host variable initialization
	int targetStringLength = CHROMOSOME_LENGTH;
	char targetString[CHROMOSOME_LENGTH];
	for(int i=0; i<CHROMOSOME_LENGTH; i++){
		targetString[i] = GENES[rand()%GENE_LENGTH];
	}
	int convergeMin[] = {GENERATIONS+1};
	int population_memory_size = POPULATION_SIZE * sizeof(Individual);
	Individual population[POPULATION_SIZE];
	
	// Random populartion initialization
	srand(time(0));
	
	for(int i=0; i<POPULATION_SIZE; i++){
		for(int j=0; j<targetStringLength; j++){
			population[i].chromosome[j] = GENES[rand()%GENE_LENGTH];
		}
	}

	// Random number generator for device
	curandState *devStates;
	CUDA_ERROR_CHECK( cudaMalloc((void **)&devStates, BLOCK_DIMENSION * GRID_DIMENSION * sizeof(curandState)) );
	setup_kernel<<<GRID_DIMENSION, BLOCK_DIMENSION>>>(devStates);
	CUDA_ERROR_CHECK( cudaPeekAtLastError() );

	// Device time measurement variables
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate memory in device
	Individual *population_d;
	char *targetString_d;
	int *convergeGen;
	
	CUDA_ERROR_CHECK( cudaMalloc((void **)&population_d, population_memory_size) );
	CUDA_ERROR_CHECK( cudaMalloc((void **)&targetString_d, targetStringLength * sizeof(char)) );
	CUDA_ERROR_CHECK( cudaMalloc((void **)&convergeGen, sizeof(int)) );

	// Copy data from host to device
	cudaEventRecord(start);	
	CUDA_ERROR_CHECK( cudaMemcpy(population_d, population, population_memory_size, cudaMemcpyHostToDevice) );
	CUDA_ERROR_CHECK( cudaMemcpy(targetString_d, targetString, targetStringLength * sizeof(char), cudaMemcpyHostToDevice) );
	CUDA_ERROR_CHECK( cudaMemcpy(convergeGen, convergeMin, sizeof(int), cudaMemcpyHostToDevice) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&h2dTime, start, stop);

	// Assign parameters and launch kernel
	cudaEventRecord(start);
	geneticAlgorithmKernel<<<GRID_DIMENSION, BLOCK_DIMENSION>>>(population_d, targetString_d, targetStringLength, devStates, convergeGen);
        CUDA_ERROR_CHECK( cudaPeekAtLastError() );
	cudaEventRecord(stop);
        cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernelTime, start, stop);

	// Fetch results back from kernel
	cudaEventRecord(start);
	CUDA_ERROR_CHECK( cudaMemcpy(population, population_d, population_memory_size, cudaMemcpyDeviceToHost) );
	CUDA_ERROR_CHECK( cudaMemcpy(convergeMin, convergeGen, sizeof(int), cudaMemcpyDeviceToHost) );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
	cudaEventElapsedTime(&d2hTime, start, stop);

	// Print output values
	printf("%d,%f,%f,%f\n", convergeMin[0], h2dTime, kernelTime, d2hTime);
	/*printf("\nAnswer obtained after %d generations\n", convergeMin[0]);
	printf("Host to device copy time:\t%f ms\n", h2dTime);
	printf("Kernel execution time:\t%f ms\n", kernelTime);
	printf("Device to host copy time:\t%f ms\n\n", d2hTime);*/

	CUDA_ERROR_CHECK( cudaFree(devStates) );
	CUDA_ERROR_CHECK( cudaFree(population_d) );
	CUDA_ERROR_CHECK( cudaFree(targetString_d) );
	
	return 0;
}
