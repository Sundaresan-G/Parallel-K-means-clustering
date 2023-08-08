#include<iostream>
#include<vector>
#include<string>
#include<sstream>
#include<utility>
#include<fstream>
#include<cstring>
#include<chrono>

#define FILENAME "mdcgenpy/100mil_k_20.csv"
#define NDIMENSIONS 6
#define KNUMBER 20
#define EPSILON 1e-6
#define MAX_ITER_COUNT 500

using namespace std;
using namespace std::chrono;

vector<vector<float>> fromFileGetPoints(ifstream &file){
    vector<vector<float>> resultPoints;
    string line;
    while(getline(file, line) && line.size()>0){
        stringstream sslines(line);
        string word;
        vector<float> pointsVector;
        while(getline(sslines, word, ',')){
            float point = stof(word);
            pointsVector.push_back(point);
        }
        resultPoints.push_back(pointsVector);
    }

    return resultPoints;
}

void convertVectorTo1DArray(float *localPointArray,\
 vector<vector<float>> localPointVectorArray, int no_of_columns){
    for(int i=0; i<localPointVectorArray.size(); i++){
        for(int j=0; j<no_of_columns; j++){
            localPointArray[i*no_of_columns + j] = localPointVectorArray[i][j];
        }
    }

}

void initialCentroidGeneration(float *centroids, int knumber, int dimensions, int seed){
    srand(seed);
    for(int i=0; i<knumber; i++){
        for(int j=0; j < dimensions; j++){
            centroids[i*dimensions + j] = (float)rand()/RAND_MAX;
        }
    }
}

extern "C" int computeKmeansCentroid(const float *localPointArray, const int num_points,\
const int dimensions, float *centroids, int *clusterSize, \
const int knumber, const int max_iter_count, const float epsilon, float *finalNormChange);
    

int main(int argc, char *argv[]){
    // argv[1], argv[2] - for seed , filename respectively
    
    char filename[500] = FILENAME;

    if(argc>2){
        string filename(argv[2]);
        // strcpy(filename, argv[2]);
    }

    ifstream f_open(filename);
    // f_open.open(filename);
    if(!f_open){
        cout<<"File could not be opened\n";
        return -1;
    }

    auto begin_withFile = high_resolution_clock::now();

    vector<vector<float>> localPointVectorArray = fromFileGetPoints(f_open);

    f_open.close();

    float *localPointArray = new float[localPointVectorArray.size()*NDIMENSIONS];
    if(localPointArray==NULL){
        cout<<"localPointArray: sufficient memory not available\n";
        return -1;
    }
    // Convert vector to 1D array
    convertVectorTo1DArray(localPointArray, localPointVectorArray, NDIMENSIONS);

    // For checking correctness
    // cout<<"The first few points are\n";
    // for(int i=0; i<10; i++){
    //     string tempfull;
    //     for(int j=0; j<NDIMENSIONS; j++){
    //         // cout<<localPointArray[i*NDIMENSIONS + j]<<'\t';
    //         char temp[20];
    //         sprintf(temp, "%8f\t", localPointArray[i*NDIMENSIONS + j]);
    //         tempfull+=string(temp);
    //     }
    //     cout<<tempfull<<endl;
    // }

    auto begin = high_resolution_clock::now();

    /* Initial guess of Centroids by CPU */

    int knumber = KNUMBER;
    float *centroids = new float[knumber*NDIMENSIONS];
    int *clusterSize = new int[knumber]{0};
    // float *newCentroids = new float[knumber*NDIMENSIONS];
 
    int seed = knumber;
    if(argc>1){
        seed = stoi(argv[1]);
    }

    initialCentroidGeneration(centroids, knumber, NDIMENSIONS, seed);

    int num_points = localPointVectorArray.size();
    

    // computeKmeansCentroid returns the iteration_count required for convergence
    int iteration_count = 0;

    // Print centroids, cluster count, final iteration number
    printf("Initial Iteration count: %d\nInitial centroids are\n", iteration_count);
    printf("C_ID\tClus_size\tCentroids\n");

    for(int i=0; i < knumber; i++){
        string tempfull;
        char k_char[12];
        sprintf(k_char, "%3d\t", i);
        char size_char[20];
        sprintf(size_char, "%9d\t", clusterSize[i]);
        cout<<k_char<<size_char;
        for(int j=0; j<NDIMENSIONS; j++){
            // cout<<localPointArray[i*NDIMENSIONS + j]<<'\t';
            char temp[20];
            sprintf(temp, "%8f\t", centroids[i*NDIMENSIONS + j]);
            tempfull+=string(temp);
        }
        cout<<tempfull<<endl;
    }

    float normCentroidChange;

    iteration_count = computeKmeansCentroid(localPointArray, num_points, NDIMENSIONS, \
    centroids, clusterSize, knumber, MAX_ITER_COUNT, EPSILON, &normCentroidChange);

    if(iteration_count==-1){
        cout<<"Kmeans function compute error";
        return -1;
    }  

    auto end = high_resolution_clock::now();

    cout<<endl;

    cout<<"The given file is: "<<filename<<endl;

    std::cout << "Time difference with file handling = " \
    << (float)duration_cast<microseconds>(end - begin_withFile).count()/1000\
     << " [ms]" << std::endl;
    std::cout << "Time difference without file handling = " \
    << (float)duration_cast<microseconds>(end - begin).count()/1000\
     << " [ms]" << std::endl<<endl;

    // Sum of clusterSize
    int sum_clusterSize=0;
    for(int i=0; i<knumber; i++){
        sum_clusterSize += clusterSize[i];
    }

    // Print centroids, cluster count, final iteration number
    printf("Final Iteration count: %d\nFinal centroids are\n\
    Total num_points: %d\n", iteration_count, sum_clusterSize);

    cout<<"Final norm of centroid change: "<<normCentroidChange<<endl;
    printf("C_ID\tClus_size\tCentroids\n");

    for(int i=0; i < knumber; i++){
        string tempfull;
        char k_char[12];
        sprintf(k_char, "%3d\t", i);
        char size_char[20];
        sprintf(size_char, "%9d\t", clusterSize[i]);
        cout<<k_char<<size_char;
        for(int j=0; j<NDIMENSIONS; j++){
            // cout<<localPointArray[i*NDIMENSIONS + j]<<'\t';
            char temp[20];
            sprintf(temp, "%8f\t", centroids[i*NDIMENSIONS + j]);
            tempfull+=string(temp);
        }
        cout<<tempfull<<endl;
    }

    delete centroids;
    // delete newCentroids;
    delete localPointArray;

    return 0;
    
}