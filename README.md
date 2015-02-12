# vector-add-opencl

OpenCL implementation of parallel vector addition.  This is a lot longer than CUDA as you need to explicitly create a context for a particular device, write a kernel in char form, explicitly compile and then instantiate the kernel, explicitly set the input and output arguments via a function, and explicitly wait for the kernel to finish.
