// MP 1
#include <wb.h>
#include <stdlib.h> // For malloc and rand
#include <time.h>   // For time


__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    out[i] = in1[i] + in2[i];
  }
}



int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  // Check if at least one argument was provided
  if (argc < 2) {
    printf("Usage: %s length\n", argv[0]);
    return 1;
  }

  // The first positional argument is at argv[1]
  int inputLength = atoi(argv[1]); // Convert the string argument to an integer

  printf("Input Length: %d\n", inputLength);

  wbTime_start(Generic, "Importing data and creating memory on host");

  // random input 
  hostInput1 = (float *)malloc(inputLength * sizeof(float));
  hostInput2 = (float *)malloc(inputLength * sizeof(float));

   // Initialize random seed
   srand(time(NULL));

   // Fill the arrays with random values
   for(int i = 0; i < inputLength; i++) {
       hostInput1[i] = (float)rand() / RAND_MAX; // Random float between 0.0 and 1.0
       hostInput2[i] = (float)rand() / RAND_MAX; // Random float between 0.0 and 1.0
   }


  // hostInput1 =
  //     (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  // hostInput2 =
  //     (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);

  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  float *A_d, *B_d, *C_d;
  int size = inputLength * sizeof(float);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
 
  cudaMalloc((void **) &A_d, size);
  cudaMalloc((void **) &B_d, size);
  cudaMalloc((void **) &C_d, size);


  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(A_d, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, hostInput2, size, cudaMemcpyHostToDevice);

  //@@ Copy memory to the GPU here

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  vecAdd<<<(inputLength+255)/256, 256>>>(A_d, B_d, C_d, inputLength);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, C_d, size, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  wbTime_stop(GPU, "Freeing GPU Memory");

  //wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  // check whether the result is correct
  for(int i = 0; i < inputLength; i++) {
    if (hostOutput[i] != hostInput1[i] + hostInput2[i]) {
      printf("Error: %f + %f != %f\n", hostInput1[i], hostInput2[i], hostOutput[i]);
      return -1;
    }
  }
  printf("All results are correct!\n");

  return 0;
}
