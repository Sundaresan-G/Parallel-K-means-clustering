# Parallel-K-means-clustering
There are three different implementations of the k means clustering algorithm:

CPU-Only Implementation: This version uses MPI and OpenMP for parallelism. It involves broadcasting centroids to all processors, and then in each iteration, a utility function is used to compute new centroids for each cluster. OpenMP is employed to parallelize the computation of cluster assignments for each data point. MPI_Allreduce is used to aggregate cluster sums and counts across all processes. `mpi_openmp_version` has the implementation.

GPU-Only Implementation: In this version, only CUDA is used for parallelism. It employs two kernels: one to assign each data point to a cluster, and another to compute new centroids based on partial sums and counts. Shared memory is used to optimize performance. `cuda` has the implementation.

Hybrid Implementation: This implementation combines MPI, OpenMP, and CUDA. It starts by broadcasting centroids to all processors and uses a utility function for centroid computation. Unlike the CPU-only version, it employs both CUDA and OpenMP for parallelism. Data points are divided between CUDA and OpenMP, and their results are combined to obtain the new centroids. `mpi_openmp_cuda` has the implementation.
