#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>
#include <sched.h>

using namespace std;

#define INFY 2000000000

#define TPB 256

int THREADS_PER_BLOCK = TPB;

float *d_localPointArray;

extern "C" void setDeviceProps(int rank, int size){
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    int device = rank % nDevices;
    cudaSetDevice(device);
}

extern "C" void getDeviceProps(int rank, int size){

    int device;

    cudaGetDevice(&device);

    int nDevices;
    cudaGetDeviceCount(&nDevices);

    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname, 1023);

    printf("My GPU device ID is: %d out of GPU devices: %d in host: %s for MPI rank : %d out of size: %d\n", device, nDevices, hostname, rank, size);

}



void convertVectorTo1DArray(float *localPointArray,\
 vector<vector<float>> localPointVectorArray, int no_of_columns){
    for(int i=0; i<localPointVectorArray.size(); i++){
        for(int j=0; j<no_of_columns; j++){
            localPointArray[i*no_of_columns + j] = localPointVectorArray[i][j];
        }
    }
}

extern "C" void cudaInitialize(vector<vector<float>> localPointArray, \
int num_points, int dimensions){

    float *Array = new float[num_points*dimensions];
    convertVectorTo1DArray(Array,\
    localPointArray, dimensions);

    cudaMalloc((void **)&d_localPointArray, num_points*dimensions*sizeof(float)); 

    // Copy from host to device localPointArray
    cudaMemcpy(d_localPointArray, Array, \
     num_points*dimensions*sizeof(float), cudaMemcpyHostToDevice);  


}

extern "C" void cudaDeInitialize(){

    cudaFree(d_localPointArray);

}

// The below function only returns sum as d_centroids and size as d_clusterSize. Do division separately
__global__ void kMeansComputeSum(float *d_localPointArray, \
int *d_clust_assn, float *d_centroids, int *d_clusterSize, int num_points, int knumber, const int dimensions){
    //get idx for this thread
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // Total size : (TPB*dimensions + knumber*dimensions + knumber + TPB) * sizeof(float)
    extern __shared__ float s[]; 

    // size TPB*dimensions
    float *shared_datapoints = (float *)&s[0];

    // size knumber*dimensions
    float *sum = (float *)&shared_datapoints[TPB*dimensions];

    // size knumber
    int *clusterNumPoints = (int *)&sum[knumber*dimensions];

    // size TPB
    int *s_clust_assn = (int *)&clusterNumPoints[knumber];
    
    //get threadID within a block. Useful later for shared memory handling
	const int s_idx = threadIdx.x;        

    // Ensure that knumber*dimensions less than THREADS_PER_BLOCK
    if (s_idx < knumber*dimensions){
        sum[s_idx] = 0;
    }

    if (s_idx < knumber){
        clusterNumPoints[s_idx] = 0;
    }

	if (idx < num_points){
        s_clust_assn[s_idx] = d_clust_assn[idx];
        for(int i=0; i<dimensions; i++){
            shared_datapoints[s_idx*dimensions + i] = d_localPointArray[idx*dimensions + i];
        }
    }    

    __syncthreads();

    // Each 6 threads in a block will sum the clusterpoints in respective dimensions
    if(s_idx < 6){
        // Except the last block
        if(blockIdx.x < gridDim.x-1){
            for(int j=0; j < blockDim.x; ++j)
            {
                int clust_id = s_clust_assn[j];
                sum[clust_id*dimensions + s_idx] += shared_datapoints[j*dimensions + s_idx];
                clusterNumPoints[clust_id] += 1;
            }
        } // For the last block
        else {
            int lastPoint = num_points % blockDim.x;
            int finalPoint = (lastPoint!=0) ? lastPoint : blockDim.x;
            for(int j=0; j < finalPoint; ++j)
            {
                int clust_id = s_clust_assn[j];
                sum[clust_id*dimensions + s_idx] += shared_datapoints[j*dimensions + s_idx];
                clusterNumPoints[clust_id] += 1;
            }
        }        

		//Now we add the sums to the global centroids and add the counts to the global counts.
		for(int z=0; z < knumber; z++)
		{
			atomicAdd(&d_centroids[z*dimensions + s_idx], sum[z*dimensions + s_idx]);
		}
    }

    __syncthreads();

    if(s_idx < knumber){
        atomicAdd(&d_clusterSize[s_idx],clusterNumPoints[s_idx]);
    }

    // __syncthreads();

    // //currently centroids are just sums, so divide by size to get actual centroids
	// if(idx < knumber*dimensions){
    //     int centroidNumber = idx/dimensions;
    //     if(d_clusterSize[centroidNumber]!=0){
    //         d_centroids[idx] = d_centroids[idx]/d_clusterSize[centroidNumber];
    //         d_centroids[idx - knumber*dimensions] = d_centroids[idx];
    //     } else {
    //         d_centroids[idx] = d_centroids[idx - knumber*dimensions];
    //     }
		 
	// }        
    
}

// We use 1-norm for the below function
__host__ __device__ float distance(float *x1, float *x2, int dimensions){
	float result = 0;
    for(int i=0; i<dimensions; i++){
        result += abs(x2[i]-x1[i]);
    }
    return result;
}  

__global__ void kMeansClusterAssignment(float *d_datapoints, \
int *d_clust_assn, float *d_centroids, int knumber, int num_points, int dimensions){

	//get idx for this thread
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // if(idx==0){
    //     printf("knumber = %d, Dimensions = %d\n", knumber, dimensions);
    // }

    // get datapoint location (starting location of data point) for each idx
    const int data_idx = idx*dimensions;

    //get threadID within a block. Useful later for shared memory handling
	const int s_idx = threadIdx.x;

    extern __shared__ float s2[];

    float *device_centroids = s2;

    // Ensure that knumber*dimensions less than THREADS_PER_BLOCK
    if (s_idx < knumber*dimensions){
        device_centroids[s_idx] = d_centroids[s_idx];
    }    

    // if (idx < knumber*dimensions){
    //     printf("Thread id/data point: %d, centroid[0]: %f\n", idx, device_centroids[s_idx]);
    // }  

    //bounds check
	if (idx >= num_points) return;

    __syncthreads();

	//find the closest centroid to this datapoint
	float min_dist = INFY;
	int closest_centroid = 0;

	for(int c = 0; c < knumber ; c++)
	{
        float dist = distance(&d_datapoints[data_idx], &device_centroids[c*dimensions], dimensions);

        if(dist < min_dist)
        {
            min_dist = dist;
            closest_centroid=c;
        }
	}

	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx]=closest_centroid;
}


// omp_start is the size i.e cuda will handle from 0 to omp_start - 1
extern "C" int computeSumCount(float *centroids, vector<vector<float>> localPointArray, int localPointArraySize, \
int omp_start, int knumber, int dimensions, float *localClusterSum, int *localClusterCount, int rank, int size){

    int openmpthreads = -1;

    float *h_cudaClusterSum = (float *)calloc(knumber*dimensions, sizeof(float));
    int *h_cudaClusterCount = (int *)calloc(knumber, sizeof(int));

    int *localClusterLabel = (int *)calloc(localPointArraySize, sizeof(int));

    int cuda_num_points = omp_start;

    int num_blocks = (cuda_num_points + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    float *d_centroids; 
    int *d_clust_assn;
    
    cudaMalloc((void **)&d_clust_assn, cuda_num_points*sizeof(int));
    
    cudaMalloc((void **)&d_centroids, knumber*dimensions*sizeof(float));

    cudaMemcpy(d_centroids, centroids,  \
        knumber*dimensions*sizeof(float), cudaMemcpyHostToDevice);    

    kMeansClusterAssignment<<<num_blocks, THREADS_PER_BLOCK,\
     knumber*dimensions*sizeof(float)>>>(d_localPointArray, \
    d_clust_assn, d_centroids, knumber, cuda_num_points, dimensions);

    #pragma omp parallel shared(localClusterLabel, centroids, localPointArray, localPointArraySize, rank, size)
    {        
        openmpthreads=omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        // printf("My OMP thread ID is: %d out of total threads: %d.\n\
        //  My CPU ID is %d. My MPI rank is %d. My details are\n", thread_id, openmpthreads, sched_getcpu(), rank);
        // cout<<"Hello from omp\n";
        #pragma omp for reduction(+:localClusterCount[:knumber], localClusterSum[:knumber*dimensions])
        for(int i = omp_start; i < localPointArraySize; i++){    
            float min_dist = 100000000; // a huge number
            for(int j=0; j < knumber; j++){
                float localDist = 0;
                for(int k=0; k< dimensions; k++){
                    localDist += fabs(localPointArray[i][k]-centroids[j * dimensions + k]);
                }
                if(localDist < min_dist){
                    min_dist = localDist;
                    localClusterLabel[i] = j;
                }
            }
            localClusterCount[localClusterLabel[i]]++;
            for(int k=0; k < dimensions; k++){
                localClusterSum[localClusterLabel[i] * dimensions + k] += localPointArray[i][k];
            }
        }
    }

    // cout<<"I have exited OMP line 283\n";

    free(localClusterLabel);

    // To store device sum
    float *d_sum;
    int *d_clusterSize;

    cudaMalloc((void **)&d_sum, knumber*dimensions*sizeof(float));
    cudaMalloc((void **)&d_clusterSize, knumber*sizeof(int));

    //reset centroids and cluster sizes (will be updated in the next kernel)
    cudaMemset(d_sum, 0.0, knumber*dimensions*sizeof(float));
    cudaMemset(d_clusterSize, 0, knumber*sizeof(int));

    int shared_mem_size = (TPB*dimensions + knumber*dimensions + knumber + TPB) * sizeof(float);

    //call centroid update kernel. Only gives sum and d_clusterSize. We need to divide separately
    kMeansComputeSum<<<num_blocks, THREADS_PER_BLOCK, shared_mem_size>>>(d_localPointArray, \
    d_clust_assn, d_sum, d_clusterSize, cuda_num_points, knumber, dimensions);  

    cudaMemcpy(h_cudaClusterCount, d_clusterSize, \
        knumber*sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(h_cudaClusterSum, d_sum, \
        knumber*dimensions*sizeof(float), cudaMemcpyDeviceToHost);

    #pragma omp parallel for
    for(int i=0; i<knumber; i++){
        localClusterCount[i] += h_cudaClusterCount[i];
    }

    #pragma omp parallel for
    for(int i=0; i<knumber*dimensions; i++){
        localClusterSum[i] += h_cudaClusterSum[i];
    }

    cudaFree(d_sum); 
    cudaFree(d_clusterSize);
    cudaFree(d_centroids); 
    cudaFree(d_clust_assn);

    return openmpthreads;
    
}
