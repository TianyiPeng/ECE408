#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> // For malloc and rand

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < numARows && j < numBColumns) {
    float sum = 0;
    for (int k = 0; k < numAColumns; k++) {
      sum += A[i * numAColumns + k] * B[k * numBColumns + j];
    }
    C[i * numCColumns + j] = sum;
  }
}

int main(int argc, char **argv) {
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;

  if (argc != 6) {
    printf("Usage: %s n block_size\n", argv[0]);
    return 1;  // Return non-zero to indicate error
}

  // Convert command-line arguments from strings to integers
  int64_t n = atoi(argv[1]);
  int64_t m = atoi(argv[2]);
  int64_t p = atoi(argv[3]);
  int block_size_x = atoi(argv[4]);
  int block_size_y = atoi(argv[5]);

  int64_t numARows = n;    // number of rows in the matrix A
  int64_t numAColumns = m; // number of columns in the matrix A
  int64_t numBRows = m;    // number of rows in the matrix B
  int64_t numBColumns = p; // number of columns in the matrix B
  //int numBColumns = 28672 / 8; // number of columns in the matrix B
  int64_t numCRows;    // number of rows in the matrix C (you have to set this)
  int64_t numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  numCRows = numARows;
  numCColumns = numBColumns;
  
  hostA = (float *)malloc(numARows * numAColumns * sizeof(float));
  hostB = (float *)malloc(numBRows * numBColumns * sizeof(float));

  // Initialize the A and B matrices
  int i1 = n-1;
  int j1 = p-1;
  for (int k = 0; k < numAColumns; k++) {
    hostA[i1 * numAColumns + k] = (float)rand() / RAND_MAX;
  }
  for (int k = 0; k < numBRows; k++) {
    hostB[k * numBColumns + j1] = (float)rand() / RAND_MAX;
  }

  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(float));

  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  struct timeval start, end;
  float kernel_time;

  gettimeofday(&start, NULL);

  dim3 gridSize(ceil((float)numARows / block_size_x), ceil((float)numBColumns / block_size_y), 1);
  dim3 blockSize(block_size_x, block_size_y, 1);

  matrixMultiply<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();

  gettimeofday(&end, NULL);

  kernel_time = end.tv_sec - start.tv_sec
      + (float) (end.tv_usec - start.tv_usec) / 1e6;

  printf("dimension of A = %lld x %lld\n", numARows, numAColumns);
  printf("dimension of B = %lld x %lld\n", numBRows, numBColumns);
  printf("time = %.5f for matrix multiplication\n", kernel_time);

  double bandwidth = ((double)numARows * numAColumns + (double)numBRows * numBColumns + (double)numCRows * numCColumns) * sizeof(float) / kernel_time / 1e9;

  printf("bandwidth = %.5lf\n", bandwidth);

  printf("tflops = %.5lf\n", (2.0 * numARows * numAColumns * numBColumns) / kernel_time / 1e12);

  // check correctness
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  
  hostC[i1 * numCColumns + j1] = 0;

  for (int k = 0; k < numAColumns; k++) {
         hostC[i1 * numCColumns + j1] += hostA[i1 * numAColumns + k] * hostB[k * numBColumns + j1];
       }
  printf("hostC[%d][%d] = %f\n", i1, j1, hostC[i1 * numCColumns + j1]);

  float *hostC2 = (float *)malloc(numCRows * numCColumns * sizeof(float));

  cudaMemcpy(hostC2, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  // check the result
  if (hostC[i1 * numCColumns + j1] != hostC2[i1 * numCColumns + j1]) {
    printf("Error: %f != %f\n", hostC[i1 * numCColumns + j1], hostC2[i1 * numCColumns + j1]);
    return -1;
  }

  // for (int i = 0; i < numCRows * numCColumns; i++) {
  //   if (hostC[i] != hostC2[i]) {
  //     printf("Error: %f != %f\n", hostC[i], hostC2[i]);
  //     return -1;
  //   }
  // }

  return 0;
}
