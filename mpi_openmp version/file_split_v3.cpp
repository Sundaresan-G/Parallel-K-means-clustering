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

#define FILENAME "mdcgenpy/1mil_k_15.csv"
#define NDIMENSIONS 6
#define KNUMBER 15
#define NTHREADS 2
#define EPSILON 1e-6
#define MAX_ITER_COUNT 10000

// "/data/k-means/dataPoints.txt"

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

    if(fileSplit(filename, localPointArray)==-1){
        cout<<"File could not be opened\n";
        return -1;
    }

    // double time_start_without_filehandling = MPI_Wtime();

    long long int localPointArraySize = localPointArray.size();

    MPI_Barrier(MPI_COMM_WORLD);

    printf("\nLocal number of lines: %lld for rank: %d\n", localPointArraySize, rank);
    
    long long int totalSize;

    MPI_Allreduce(&localPointArraySize, \
    &totalSize, 1, MPI_LONG_LONG_INT, \
    MPI_SUM, MPI_COMM_WORLD);

    printf("Total number of lines: %lld for rank: %d\n", totalSize, rank);

    MPI_Barrier(MPI_COMM_WORLD);

    double time_end = MPI_Wtime();

    double final_time_start_with_filehandling, final_time_start_without_filehandling;

    // MPI_Allreduce(&time_start_without_filehandling, \
    &final_time_start_without_filehandling, 1, MPI_DOUBLE, \
    MPI_MIN, MPI_COMM_WORLD);

    MPI_Allreduce(&time_start_with_filehandling, \
    &final_time_start_with_filehandling, 1, MPI_DOUBLE, \
    MPI_MIN, MPI_COMM_WORLD);

    double duration_with_file = time_end - final_time_start_with_filehandling;
    // double duration_without_file = time_end - final_time_start_without_filehandling;

    if(rank==0){

        cout<<"MPI Processes: "<<size<<endl;

        cout<<"Total time with file handling: "<<duration_with_file*1000<<\
        " ms\n";        

    }

    // printf("Hello from %d on %s out of %d\n", rank, processor_name, size);

    MPI_Finalize();
    
    return 0;

}