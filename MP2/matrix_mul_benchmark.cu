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

  if (argc != 3) {
    printf("Usage: %s n block_size\n", argv[0]);
    return 1;  // Return non-zero to indicate error
}

  // Convert command-line arguments from strings to integers
  int n = atoi(argv[1]);
  int block_size = atoi(argv[2]);

  int numARows = n;    // number of rows in the matrix A
  int numAColumns = n; // number of columns in the matrix A
  int numBRows = n;    // number of rows in the matrix B
  int numBColumns = n; // number of columns in the matrix B
  //int numBColumns = 28672 / 8; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  numCRows = numARows;
  numCColumns = numBColumns;
  
  hostA = (float *)malloc(numARows * numAColumns * sizeof(float));
  hostB = (float *)malloc(numBRows * numBColumns * sizeof(float));

  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(float));

  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  struct timeval start, end;
  float kernel_time;

  gettimeofday(&start, NULL);

  matrixMultiply<<<ceil(numBColumns/block_size),block_size>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();

  gettimeofday(&end, NULL);

  kernel_time = end.tv_sec - start.tv_sec
      + (float) (end.tv_usec - start.tv_usec) / 1e6;

  printf("dimension of A = %d x %d\n", numARows, numAColumns);
  printf("dimension of B = %d x %d\n", numBRows, numBColumns);
  printf("time = %.2f for matrix multiplication\n", kernel_time);

  double bandwidth = (numARows * numAColumns + numBRows * numBColumns + numCRows * numCColumns) * sizeof(float) / kernel_time / 1e9;

  printf("bandwidth = %.2lf\n", bandwidth);

  // check correctness
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  for (int i = 0; i < numCRows * numCColumns; i++) {
    hostC[i] = 0;
  }
  // for (int i = 0; i < numARows; i++) {
  //   for (int j = 0; j < numBColumns; j++) {
  //     for (int k = 0; k < numAColumns; k++) {
  //       hostC[i * numCColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
  //     }
  //   }
  // }

  float *hostC2 = (float *)malloc(numCRows * numCColumns * sizeof(float));

  cudaMemcpy(hostC2, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  // for (int i = 0; i < numCRows * numCColumns; i++) {
  //   if (hostC[i] != hostC2[i]) {
  //     printf("Error: %f != %f\n", hostC[i], hostC2[i]);
  //     return -1;
  //   }
  // }

  return 0;
}
