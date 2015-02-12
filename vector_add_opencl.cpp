#include <wb.h> //@@ wb include opencl.h for you

#define clCheckPtr(err, stmt) do {                                 \
  err = CL_SUCCESS;\
  stmt;                            \
  if (err != CL_SUCCESS) {                          \
    wbLog(ERROR, "Failed to run stmt ", #stmt, " with error ", err);    \
    return -1;                                     \
  }                                                  \
} while(0)

#define clCheck(stmt) do {                                 \
  cl_int err = stmt;                            \
  if (err != CL_SUCCESS) {                          \
    wbLog(ERROR, "Failed to run stmt ", #stmt, " with error ", err);    \
    return -1;                                     \
  }                                                  \
} while(0)

#define NUM_THREADS 256

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  cl_mem deviceInput1;
  cl_mem deviceInput2;
  cl_mem deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  //@@ create context
  //prepare properties
  cl_uint numPlatforms;
  cl_int clerr = clGetPlatformIDs(0, NULL, &numPlatforms);
  cl_platform_id platforms[numPlatforms];
  clerr = clGetPlatformIDs(numPlatforms, platforms, NULL);
  cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[0], 0 };
  cl_context clctx;
  //actually make the context
  clCheckPtr(clerr, clctx=clCreateContextFromType(properties,CL_DEVICE_TYPE_ALL,NULL,NULL, &clerr));

  //@@ create command queue for a device in context.
  size_t paramsz;
  cl_command_queue clcmdq;
  clCheck(clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, ¶msz)); //get the size
  cl_device_id* cldevs = (cl_device_id *) malloc(paramsz);
  clCheck(clGetContextInfo(clctx, CL_CONTEXT_DEVICES, paramsz, cldevs, NULL)); //put list of devices in cldevs

  clCheckPtr(clerr, clcmdq = clCreateCommandQueue(clctx, cldevs[0], 0, &clerr)); //explicitly create queue for device 0 so
  //we have a place to tell it to do stuff.

  //@@ OpenCL kernel char creation and compilation
  const char* vaddsrc = 
    "__kernel void vadd(__global const float *a, __global const float *b, __global float *result, int N){"\
    " int id = get_global_id(0);"\
    " if (id < N)" \ 
    "   result[id] = a[id] + b[id];"\
    "}";    
  cl_program clpgm;
  cl_kernel clkern;
  clCheckPtr(clerr, clpgm = clCreateProgramWithSource(clctx, 1, &vaddsrc, NULL, &clerr));
  char clcompileflags[4096];
  sprintf(clcompileflags, "-cl-mad-enable"); //activate multiply add on openCL, replaces a*b+c with specialized mad operation.
  clCheck(clBuildProgram(clpgm, 0, NULL, clcompileflags, NULL, NULL)); //actually compile the program
  clCheckPtr(clerr, clkern = clCreateKernel(clpgm, "vadd", &clerr)); //actually make the kernel exist.


  wbTime_start(GPU, "Allocating and copying memory to GPU.");
  //@@ Allocate and copy GPU memory here
  //note: can do clCreateBuffer with MEM_READ_ONLY followed by a clEnqueueWriteBuffer for transfer instead.
  //will do both at once though since this code is long enough.
  clCheckPtr(clerr, deviceInput1 = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLength * sizeof(float),
        hostInput1, &clerr));
  clCheckPtr(clerr, deviceInput2 = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLength * sizeof(float),
        hostInput2, &clerr));
  clCheckPtr(clerr, deviceOutput = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY, inputLength * sizeof(float),
        NULL, &clerr)); //no transfer is needed for answer as we derive that later.

  wbTime_stop(GPU, "Allocating and copying memory to the GPU.");


  //@@ set kernel arguments
  clCheck(clSetKernelArg(clkern, 0, sizeof(cl_mem), (void *) &deviceInput1));
  clCheck(clSetKernelArg(clkern, 1, sizeof(cl_mem), (void *) &deviceInput2));
  clCheck(clSetKernelArg(clkern, 2, sizeof(cl_mem), (void *) &deviceOutput));
  clCheck(clSetKernelArg(clkern, 3, sizeof(int), &inputLength));


  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch NDRangeKernel and wait for completion
  cl_event event = NULL;
  size_t global_size = NUM_THREADS * ( (inputLength - 1)/NUM_THREADS + 1 );
  size_t local_size = NUM_THREADS;
  clCheck(clEnqueueNDRangeKernel(clcmdq, clkern, 1, NULL, &global_size, &local_size, 0, NULL, &event));
  clCheck(clWaitForEvents(1, &event));
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  clCheck(clEnqueueReadBuffer(clcmdq, deviceOutput, CL_TRUE, 0, inputLength*sizeof(float), hostOutput, 0, NULL, NULL));

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  clReleaseMemObject(deviceInput1);
  clReleaseMemObject(deviceInput2);
  clReleaseMemObject(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
