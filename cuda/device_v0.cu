#include <stdlib.h>
#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define INFY 2000000000

#define TPB 256

int THREADS_PER_BLOCK = TPB;

// The below function only returns sum as d_centroids and size as d_clusterSize. Do division separately
__global__ void kMeansCentroidUpdate(float *d_localPointArray, \
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

__global__ void divideSumByCount(float *d_centroids, \
float *d_sum, int *d_clusterSize, int knumber, int dimensions){

    extern __shared__ int s3[];

    int *shared_clusterSize = (int *)s3;

    float *shared_sum = (float *)&shared_clusterSize[knumber];

    //get idx for this thread
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    //get threadID within a block. Useful later for shared memory handling
	const int s_idx = threadIdx.x;

    if(s_idx<knumber){
        shared_clusterSize[s_idx] = d_clusterSize[s_idx];
    }

    if(s_idx<knumber*dimensions){
        shared_sum[s_idx] = d_sum[s_idx];
    }

    __syncthreads();

    if(idx<knumber*dimensions && shared_clusterSize[idx/dimensions]!=0){
        d_centroids[idx] =  shared_sum[idx]/shared_clusterSize[idx/dimensions];
    }

}

void computeNewCentroids(float *d_localPointArray, \
int num_points, int dimensions, int knumber, \
float *d_centroids, int *d_clusterSize, float *h_newCentroids, int *d_clust_assn){

    int num_blocks = (num_points + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    // Assign clusters to each point using the below kernel function
    kMeansClusterAssignment<<<num_blocks, THREADS_PER_BLOCK, knumber*dimensions*sizeof(float)>>>(d_localPointArray, \
    d_clust_assn, d_centroids, knumber, num_points, dimensions);

    // To store device sum
    float *d_sum;
    cudaMalloc((void **)&d_sum, knumber*dimensions*sizeof(float));

    //reset centroids and cluster sizes (will be updated in the next kernel)
    cudaMemset(d_sum, 0.0, knumber*dimensions*sizeof(float));
    cudaMemset(d_clusterSize, 0, knumber*sizeof(int));

    int shared_mem_size = (TPB*dimensions + knumber*dimensions + knumber + TPB) * sizeof(float);

    //call centroid update kernel. Only gives sum. We need to divide separately
    // Below later half of centroids are sent to retain old values too for comparison
    kMeansCentroidUpdate<<<num_blocks, THREADS_PER_BLOCK, shared_mem_size>>>(d_localPointArray, \
    d_clust_assn, d_sum, d_clusterSize, num_points, knumber, dimensions);  

    int req_threads = knumber*dimensions;
    int req_blocks = (req_threads + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    divideSumByCount<<<req_blocks, THREADS_PER_BLOCK, \
    knumber*sizeof(int) + knumber*dimensions*sizeof(float) >>>(d_centroids, \
    d_sum, d_clusterSize, knumber, dimensions); 

    cudaFree(d_sum); 

    //copy new centroids back to host 
    cudaMemcpy(h_newCentroids, d_centroids, knumber*dimensions*sizeof(float),cudaMemcpyDeviceToHost);

}

extern "C" int computeKmeansCentroid(const float *localPointArray, const int num_points,\
const int dimensions, float *centroids, int *clusterSize, const int knumber, const int max_iter_count, const float epsilon){
    // Variables for device memory of localPointArray and centroids
    float *d_localPointArray, *d_centroids; 
    int *d_clusterSize;
    int *d_clust_assn;
    
    cudaMalloc((void **)&d_clust_assn, num_points*sizeof(int));
    cudaMalloc((void **)&d_localPointArray, num_points*dimensions*sizeof(float));

    
    cudaMalloc((void **)&d_centroids, knumber*dimensions*sizeof(float));
    cudaMalloc((void **)&d_clusterSize, knumber*sizeof(int));

    // Copy from host to device localPointArray
    cudaMemcpy(d_localPointArray, localPointArray, \
     num_points*dimensions*sizeof(float), cudaMemcpyHostToDevice);

    float *newCentroids = new float[knumber*dimensions];

    int iterationCount = 0;
    float normCentroidChange = 0;

    do {

        // Print centroids, cluster count, final iteration number
        // printf("Iteration count: %d\nCentroids are\n", iterationCount);
        // printf("C_ID\tClus_size\tCentroids\n");

        // for(int i=0; i < knumber; i++){
        //     string tempfull;
        //     char k_char[12];
        //     sprintf(k_char, "%3d\t", i);
        //     char size_char[20];
        //     sprintf(size_char, "%9d\t", clusterSize[i]);
        //     cout<<k_char<<size_char;
        //     for(int j=0; j<dimensions; j++){
        //         // cout<<localPointArray[i*dimensions + j]<<'\t';
        //         char temp[20];
        //         sprintf(temp, "%8f\t", centroids[i*dimensions + j]);
        //         tempfull+=string(temp);
        //     }
        //     cout<<tempfull<<endl;
        // }

        cudaMemcpy(d_centroids, centroids,  \
        knumber*dimensions*sizeof(float), cudaMemcpyHostToDevice);

        computeNewCentroids(d_localPointArray, num_points, dimensions, \
        knumber, d_centroids, d_clusterSize, newCentroids, d_clust_assn);

        // Calculate change in centroids norm
        normCentroidChange = 0;
        for(int k=0; k<knumber; k++){
            normCentroidChange += distance(&newCentroids[k*dimensions], &centroids[k*dimensions], dimensions);
            for(int j=0; j<dimensions; j++){
                centroids[k*dimensions + j] = newCentroids[k*dimensions + j];
            }
        }

        // cout<<"Norm centroid change: "<<normCentroidChange<<endl<<endl;

        // cudaMemcpy(centroids, d_centroids, \
        // knumber*dimensions*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(clusterSize, d_clusterSize, \
        knumber*sizeof(int), cudaMemcpyDeviceToHost);

        iterationCount++;

    } while(normCentroidChange > epsilon && iterationCount<=max_iter_count);

    // cudaMemcpy(centroids, d_centroids, \
    // knumber*dimensions*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(clusterSize, d_clusterSize, \
    knumber*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_localPointArray);
    cudaFree(d_centroids);
    cudaFree(d_clusterSize);
    cudaFree(d_clust_assn);

    delete newCentroids;
    return iterationCount;    
}
