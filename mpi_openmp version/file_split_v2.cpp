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

#define FILENAME "/data/k-means/dataPoints.txt"
#define NDIMENSIONS 6
#define KNUMBER 15
#define NTHREADS 2
#define EPSILON 1e-6
#define MAX_ITER_COUNT 10000

using namespace std;

int openmpthreads;

void fileSplit(char *filename, vector<vector<float>> &localResult){ 

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

    if(fileSize%size!=0)
        mySize++;

    // cout<<mySize<<endl;

    MPI_Offset globalStart = rank*mySize;
    MPI_Offset globalEnd = globalStart + mySize - 1;    // inclusive of end

    if(rank == size-1)
        globalEnd = fileSize-1;

    int overlap = 160; //assuming '\n' occurs within 200 characters
    if(rank!=0)
        globalStart -= overlap; 

    globalStart = (globalStart>=0)? globalStart : 0;

    globalEnd = (globalEnd < fileSize)? globalEnd : fileSize-1;
    
    mySize = globalEnd - globalStart + 1;

    // An attempt. Delete later
    // mySize /= 2;
    // globalStart = rank*mySize;
    // globalEnd = globalStart + mySize - 1;

    char *chunk = new char[mySize+1]{0};
    // chunk = (char *)malloc((mySize+1)*sizeof(char));

    int buf_start = 0;
    int buf_length = 100000; // read 1000 characters everytime

    MPI_Offset no_of_iter = mySize/buf_length; //we need one extra after this

    // MPI_File_read_at_all(fh, globalStart, chunk, mySize, MPI_CHAR, MPI_STATUS_IGNORE);
    
    // MPI_File_read_at_all_begin(fh, globalStart, chunk, mySize, MPI_CHAR);

    // MPI_File_read_at_all_end(fh, chunk, MPI_STATUS_IGNORE);

    cout<<"Max no:"<<no_of_iter<<" Rank: "<<rank<<endl;
    // MPI_Request req;
    
    if(2*buf_length < mySize){
        for(int i=0; i < no_of_iter-1; i++){
            MPI_File_read_at_all(fh, globalStart+buf_start, &chunk[buf_start], buf_length, MPI_CHAR, MPI_STATUS_IGNORE);
            // MPI_Wait(&req, MPI_STATUS_IGNORE);
            buf_start += buf_length;
            if(rank==0 && i%10000 == 0){
                cout<<"Max no:"<<no_of_iter<<" Iteration: "<<i<<endl;
                cout<<"File size: "<<fileSize<<" Local size: "<<mySize<<endl<<endl;
                for(int i=0; i<100; i++){
                    cout<<chunk[buf_start-buf_length+i];
                }
                cout<<endl;
            }
        }
    }
    

    // while(buf_start + buf_length >= mySize){

    //     // if(buf_start + 2*buf_length >= mySize){
    //     //     buf_length = (buf_start + buf_length >= mySize)? mySize - buf_start : buf_length;            
    //     //     MPI_File_read_at_all(fh, globalStart+buf_start, &chunk[buf_start], buf_length, MPI_CHAR, MPI_STATUS_IGNORE);
    //     //     buf_start += buf_length;
    //     //     break;
    //     // }

    //     MPI_File_read_at(fh, globalStart+buf_start, &chunk[buf_start], buf_length, MPI_CHAR, MPI_STATUS_IGNORE);
    //     buf_start += buf_length;
    // }    

    buf_length = mySize - buf_start;            
    MPI_File_read_at_all(fh, globalStart+buf_start, &chunk[buf_start], buf_length, MPI_CHAR, MPI_STATUS_IGNORE);
    // MPI_Wait(&req, MPI_STATUS_IGNORE);
    
    
    chunk[mySize] = '\0';

    if(rank==1){
        cout<<"File size: "<<fileSize<<" Local size: "<<mySize<<endl<<endl;
        for(int i=0; i<100; i++){
            cout<<chunk[i];
        }
        cout<<endl;
    }

    //cout<<"No problem till line 55\n";
    
    MPI_Offset locstart=0, locend=mySize-1;
    if (rank != 0) {
        for(int i = 0; i < overlap; i++){
            if(chunk[i] == '\n') {
                locstart = i;
            }
        }
        if(chunk[locstart]!='\n'){
            cout<<"No line break in start\n";
        }
        locstart++;
        globalStart += locstart;
    }

    if (rank != size-1) {

        int count = 0;
        
        // globalEnd -= overlap-1;
        // locend -= overlap-1;
        // int location;
        // for(int i = 0; i < overlap; i++){
        //     if(chunk[locend + i] == '\n') {
        //         location = i;
        //     }
        // }
        // locend += location;
        // globalEnd += location;

        while(chunk[locend] != '\n' && count < overlap) {
            locend--;
            globalEnd--;
            count++;
        }

        if(chunk[locend]!='\n'){
            cout<<"No line break in end\n";
        }
    }
    
    mySize = locend-locstart+1;

    // free(chunk);
    
    // //At this point, each processor has globalStart as starting point, globalEnd as end point of file to access
    
    // chunk = (char *)malloc((mySize+1)*sizeof(char));
    // MPI_File_read_at_all(fh, globalStart, chunk, mySize, MPI_CHAR, MPI_STATUS_IGNORE);
    // chunk[mySize] = '\0';
    //cout<<chunk<<endl;    

    vector<vector<float>> localPointArray; 

    char *final_chunk = new char[mySize+1];
    final_chunk[mySize] = '\0';
    memcpy(final_chunk, &chunk[locstart], mySize); 

    delete[] chunk;   

    stringstream ss(final_chunk);
    string lines;

    cout<<"Reading from file completed and is in chunk\n";

    while(getline(ss,lines,'\n')){        
        stringstream sslines(lines);
        string word;
        vector<float> pointsVector;
        while(getline(sslines, word, ',')){
            float point = stof(word);
            pointsVector.push_back(point);
        }
        if(pointsVector.size()!=NDIMENSIONS+1){
            continue;
        }
        localPointArray.push_back(pointsVector);
    }
    
    delete[] final_chunk;

    localResult = localPointArray;

    
    // if(rank == 0){
    //     cout<<"Process rank: "<<rank<<", local number of points: ,"<<localPointArray.size()<<endl;
    //     cout<<"Local array elements are\n";

    //     for(int i=0; i<localPointArray.size(); i++){
    //         for(int j=0; j<NDIMENSIONS-1; j++){
    //             cout<<localPointArray[i][j]<<", ";
    //         }
    //         cout<<localPointArray[i][NDIMENSIONS-1]<<endl;
    //     }

    // }

    MPI_File_close(&fh);

}

int main(int argc, char *argv[]){
    // argv[1], argv[2] - for seed , filename respectively

    char filename[500] = FILENAME;

    if(argc>2){
        strcpy(filename, argv[2]);
    }

    ifstream f_open(filename);
    // f_open.open(filename);
    if(!f_open){
        cout<<"File could not be opened\n";
        return -1;
    }

    f_open.close();

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

    fileSplit(filename, localPointArray);

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