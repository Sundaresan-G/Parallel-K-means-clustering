#include<iostream>
#include<vector>
#include<string>
#include<cstring>
#include<sstream>
#include<utility>
#include<fstream>
#include<cmath>
#include "mpi.h"
#include "omp.h"

#define FILENAME "mdcgenpy/1bil_k_15.csv"
#define MAX_EXEC_COUNT 5
#define NDIMENSIONS 6
#define KNUMBER 15
#define EPSILON 1e-5
#define MAX_ITER_COUNT 100

using namespace std;

int openmpthreads;

int fileSplit(char *filename, vector<vector<float>> &localResult){ 

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);   

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    /*Distribute file among processes approximately equally*/
    MPI_Offset fileSize;
    MPI_File_get_size(fh, &fileSize);
    // fileSize--;//to remove EOF
    //cout<<fileSize<<endl;

    MPI_Offset mySize = (MPI_Offset)fileSize/size;

    MPI_File_close(&fh);

    fstream my_file;
	my_file.open(filename, ios::in);

    if(!my_file){
        return -1;
    }

    // cout<<mySize<<endl;

    MPI_Offset globalStart = rank*mySize;
    MPI_Offset globalEnd = globalStart + mySize - 1;    // inclusive of end

    if(rank == size-1)
        globalEnd = fileSize-1;

    int overlap = 160; //assuming '\n' occurs within 200 characters

    if(rank != size - 1){

        my_file.seekg(globalEnd, ios::beg);
        
        char text[500];
        overlap = (globalEnd + overlap < fileSize)? overlap: fileSize - 1 - globalEnd; 
        my_file.read(text, overlap);
        int i;
        for(i = 0; i < overlap && text[i]!='\n' && globalEnd < fileSize; i++){
            globalEnd++;
        }
        if(i==overlap || globalEnd == fileSize){
            cout<<"Error: No line break at end\n";
        }

    }

    // if(1){
    //     cout<< " Rank: "<<rank<<" Global start "<<globalStart<<endl;
    //     cout<< " Rank: "<<rank<<" Global end "<<globalEnd<<endl;
    // }

    if(rank!=0){

        globalStart--;
        my_file.seekg(globalStart, ios::beg);
        
        char text[500];
        my_file.read(text, overlap);
        int i;
        for(i = 0; i< overlap && text[i]!='\n' && globalStart < globalEnd; i++){
            globalStart++;
        }
        globalStart = (globalStart < fileSize)? globalStart + 1 : fileSize-1;
        if(i==overlap){
            cout<<"Error: No line break at start\n";
        }
    }

    // if(1){
    //     cout<< " Rank: "<<rank<<" Global start "<<globalStart<<endl;
    //     cout<< " Rank: "<<rank<<" Global end "<<globalEnd<<endl;
    // }

    mySize = globalEnd - globalStart + 1;    

    my_file.seekg(globalStart, ios::beg);

    // To count number of characters
    MPI_Offset count = 0;
    string line;
    while(getline(my_file, line) && line.size()>0 && count + line.size() + 1 <= mySize){
        stringstream sslines(line);
        string word;
        vector<float> pointsVector;
        while(getline(sslines, word, ',')){
            float point = stof(word);
            pointsVector.push_back(point);
        }
        localResult.push_back(pointsVector);
        count += line.size() + 1;
    }

    my_file.close();
    
    // To print the points
    // if(1){
    //     cout<<"Process rank: "<<rank<<", local number of points: ,"<<localResult.size()<<endl;
    //     cout<<"Local array elements are\n";

    //     for(int i=0; i<localResult.size(); i++){
    //         for(int j=0; j<NDIMENSIONS-1; j++){
    //             cout<<localResult[i][j]<<", ";
    //         }
    //         cout<<localResult[i][NDIMENSIONS-1]<<endl;
    //     }

    // }   

    return 0; 

}


void computeNewCentroids(vector<vector<float>> localPointArray, \
int localPointArraySize, float *centroids, float *newCentroids, \
int *localClusterLabel, int *newCount, int iterationCount){

    int knumber = KNUMBER;

    // Below are counts for a given process, i.e. sum of all threads
    int *globalThreadCount = (int *)calloc(knumber, sizeof(int));
    float *globalThreadSum = (float *)calloc(knumber*NDIMENSIONS, sizeof(float));

    #pragma omp parallel shared(localClusterLabel, centroids, localPointArray, localPointArraySize)
    {        
        openmpthreads=omp_get_num_threads();
        // cout<<"Hello from omp\n";
        #pragma omp for reduction(+:globalThreadCount[:knumber], globalThreadSum[:knumber*NDIMENSIONS])
        for(int i=0; i < localPointArraySize; i++){    
            float min_dist = 100000000; // a huge number
            for(int j=0; j < knumber; j++){
                float localDist = 0;
                for(int k=0; k< NDIMENSIONS; k++){
                    localDist += fabs(localPointArray[i][k]-centroids[j * NDIMENSIONS + k]);
                }
                if(localDist < min_dist){
                    min_dist = localDist;
                    localClusterLabel[i] = j;
                }
            }
            globalThreadCount[localClusterLabel[i]]++;
            for(int k=0; k < NDIMENSIONS; k++){
                globalThreadSum[localClusterLabel[i] * NDIMENSIONS + k] += localPointArray[i][k];
            }
        }
    }

    // for(int k=0; k<knumber; k++){
    //     if(globalThreadCount[k]!=0){
    //         for(int j=0; j<NDIMENSIONS; j++){
    //             globalThreadSum[knumber*NDIMENSIONS + j] /= globalThreadCount[k];
    //         }
    //     }        
    // }
    // //  Now globalThreadSum is the new centroid

    MPI_Allreduce(&globalThreadSum[0], &newCentroids[0], knumber * NDIMENSIONS, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&globalThreadCount[0], &newCount[0], knumber, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // If count!=0, update centroid else use the old value

    for(int k=0; k<knumber; k++){
        if(newCount[k]!=0){
            for(int j=0; j<NDIMENSIONS; j++){
                newCentroids[k*NDIMENSIONS + j] /= newCount[k];
            }
        }  else {
            for(int j=0; j<NDIMENSIONS; j++){
                newCentroids[k*NDIMENSIONS + j] = centroids[k*NDIMENSIONS + j];
            }
        }      
    }
    // Now newCentroids is the new centroid

    // int rank = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if(iterationCount%1==0 && rank==2){
    //     cout<<"Current iteration is "<<iterationCount<<endl;
    //     cout<<"Global count is \n";
    //     for(int i=0; i<knumber; i++){
    //         cout<<newCount[i]<<endl;
    //     }
    //     cout<<"New centroids\n";
    //     for(int j=0; j< knumber; j++){
    //         for(int k=0; k < NDIMENSIONS-1; k++){
    //             cout<<newCentroids[j * NDIMENSIONS + k]<<", ";
    //         }
    //         cout<<newCentroids[j * NDIMENSIONS + NDIMENSIONS-1]<<endl;
    //     }
    // }

    free(globalThreadCount);
    free(globalThreadSum);

}

int main(int argc, char *argv[]){
    // argv[1], argv[2] - for seed , filename respectively

    char filename[500] = FILENAME;

    if(argc>2){
        strcpy(filename, argv[2]);
    }

    //cout<<"No problem till line 16\n";
    
    MPI_Init(&argc,&argv);

    //cout<<"No problem till line 18\n";

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Splitting points among processes
    vector<vector<float>> localPointArray;
    
    // cout<<"Current filename is "<<filename<<endl;

    double time_start_with_filehandling = MPI_Wtime();

    if(rank == 0){
        cout<<"Given file is: "<<filename<<endl;
    }

    if(fileSplit(filename, localPointArray)==-1){
        if(rank == 0){
            cout<<"File could not be opened\n";
        }
        
        return -1;
    }

    int execution_count = 0;
    
    while(execution_count < MAX_EXEC_COUNT){

        double time_start_without_filehandling = MPI_Wtime();

        int localPointArraySize = localPointArray.size();

        // To get processor details to find optimum number of openmp threads
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int namelen;
        MPI_Get_processor_name(processor_name, &namelen);
        // printf("Hello from %d on %s out of %d\n", rank, processor_name, size);

        // Print Points array
        // for(int k=0; k<size; k++){
        //     // MPI_Barrier(MPI_COMM_WORLD);
        //     if(rank == k){
        //         cout<<"Process rank: "<<rank<<", local number of points: "<<localPointArraySize<<endl;
        //         cout<<"Local array elements are\n";

        //         for(int i=0; i<localPointArraySize; i++){
        //             for(int j=0; j<NDIMENSIONS; j++){
        //                 cout<<localPointArray[i][j]<<", ";
        //             }
        //             cout<<localPointArray[i][NDIMENSIONS]<<endl;
        //         }
        //     }
        //     // cout<<"My rank is "<<rank<<". My host is "<<processor_name<<" and current iteration is "<<k<<endl;
        //     // MPI_Barrier(MPI_COMM_WORLD);

        // }   

        // MPI_Barrier(MPI_COMM_WORLD);

        /* Initial guess of Centroids by rank 0 */

        int knumber = KNUMBER;
        float *centroids = (float *)malloc(knumber*NDIMENSIONS*sizeof(float));

        int seed = knumber;
        if(argc>1){
            seed = stoi(argv[1]);
        }

        if(rank == 0){
            srand(seed);
            for(int i=0; i<knumber; i++){
                for(int j=0; j < NDIMENSIONS; j++){
                    centroids[i*NDIMENSIONS + j] = (float)rand()/RAND_MAX;
                }
            }
        }

        MPI_Bcast(centroids, knumber*NDIMENSIONS, MPI_FLOAT, 0, MPI_COMM_WORLD);

        int *localClusterLabel = (int *)calloc(localPointArraySize, sizeof(int));

        float *newCentroids = (float *)malloc(knumber * NDIMENSIONS * sizeof(float));
        int *clusterCount = (int *)calloc(knumber, sizeof(int));
        float normCentroidChange;
        float epsilon = EPSILON;
        int iterationCount = 0;

        // A print of initial centroids
        if(rank==0){

            printf("Initial Iteration count: %d\nInitial centroids are\n", iterationCount);
            printf("C_ID\tClus_size\tCentroids\n");

            for(int i=0; i < knumber; i++){
                string tempfull;
                char k_char[12];
                sprintf(k_char, "%3d\t", i);
                char size_char[20];
                sprintf(size_char, "%9d\t", clusterCount[i]);
                cout<<k_char<<size_char;
                for(int j=0; j<NDIMENSIONS; j++){
                    // cout<<localPointArray[i*NDIMENSIONS + j]<<'\t';
                    char temp[20];
                    sprintf(temp, "%8f\t", centroids[i*NDIMENSIONS + j]);
                    tempfull+=string(temp);
                }
                cout<<tempfull<<endl;
            }

            // cout<<"My rank is "<<rank<<" and initial centroids are\n";
            // for(int i=0; i<knumber; i++){
            //     for(int j=0; j<NDIMENSIONS-1; j++){
            //         cout<<centroids[i*NDIMENSIONS + j]<<", ";
            //     }            
            //     cout<<centroids[i*NDIMENSIONS + NDIMENSIONS-1]<<endl;
            // }
            // cout<<endl;
        }
        
        do {

            computeNewCentroids(localPointArray, localPointArraySize, \
            centroids, newCentroids, localClusterLabel, clusterCount, iterationCount);

            // Calculate change in centroids norm
            normCentroidChange = 0;
            for(int k=0; k<knumber; k++){
                for(int j=0; j<NDIMENSIONS; j++){
                    normCentroidChange += fabs(newCentroids[k*NDIMENSIONS + j] - centroids[k*NDIMENSIONS + j]);
                    centroids[k*NDIMENSIONS + j] = newCentroids[k*NDIMENSIONS + j];
                }
            }

            iterationCount++;

        } while(normCentroidChange > epsilon && iterationCount <= MAX_ITER_COUNT);

        MPI_Barrier(MPI_COMM_WORLD);

        double time_end = MPI_Wtime();

        double final_time_start_with_filehandling, final_time_start_without_filehandling;

        MPI_Allreduce(&time_start_without_filehandling, \
        &final_time_start_without_filehandling, 1, MPI_DOUBLE, \
        MPI_MIN, MPI_COMM_WORLD);

        MPI_Allreduce(&time_start_with_filehandling, \
        &final_time_start_with_filehandling, 1, MPI_DOUBLE, \
        MPI_MIN, MPI_COMM_WORLD);

        double duration_with_file = time_end - final_time_start_with_filehandling;
        double duration_without_file = time_end - final_time_start_without_filehandling;

        execution_count++;
        
        if(rank==0){

            int sum=0;
            for(int k=0; k<knumber; k++){
                sum+=clusterCount[k];
            }

            cout<<"Current Execution Count is :"<<execution_count<<endl<<endl;

            cout<<"MPI Processes: "<<size<<", OpenMP threads: "<<openmpthreads<<endl;

            cout<<"Given file is: "<<filename<<endl;

            cout<<"Total time with file handling: "<<duration_with_file*1000<<\
            " ms, without file handling: "<<duration_without_file*1000<<" ms\n";
            
            // cout<<"My rank is "<<rank<<" and my values are\n";
            cout<<"Total number of cluster points= "<<sum<<endl;

            cout<<"Final norm of centroid change: "<<normCentroidChange<<endl;
            // cout<<"Cluster labels\n";
            // cout<<"Actual\tComputed\n";
            // for(int i=0; i<localPointArraySize && i<20; i++){
            //     cout<<localPointArray[i][NDIMENSIONS]<<'\t'<<localClusterLabel[i]<<endl;
            // }

            // Print centroids, cluster count, final iteration number
            printf("Final Iteration count: %d\nFinal centroids are\n", iterationCount);
            
            printf("C_ID\tClus_size\tCentroids\n");

            for(int i=0; i < knumber; i++){
                string tempfull;
                char k_char[12];
                sprintf(k_char, "%3d\t", i);
                char size_char[20];
                sprintf(size_char, "%9d\t", clusterCount[i]);
                cout<<k_char<<size_char;
                for(int j=0; j<NDIMENSIONS; j++){
                    // cout<<localPointArray[i*NDIMENSIONS + j]<<'\t';
                    char temp[20];
                    sprintf(temp, "%8f\t", newCentroids[i*NDIMENSIONS + j]);
                    tempfull+=string(temp);
                }
                cout<<tempfull<<endl;
            }


            // cout<<"Cluster label \t Cluster count \t Cluster centroid\n";
            // for(int j=0; j< knumber; j++){
            //     cout<<j<<"\t\t"<<clusterCount[j]<<"\t\t";
            //     for(int k=0; k < NDIMENSIONS-1; k++){
            //         cout<<newCentroids[j * NDIMENSIONS + k]<<", ";
            //     }
            //     cout<<newCentroids[j * NDIMENSIONS + NDIMENSIONS-1]<<endl;
            // }
            
        }
    }

    // printf("Hello from %d on %s out of %d\n", rank, processor_name, size);

    MPI_Finalize();
    
    return 0;

}