#define _CRT_SECURE_NO_WARNINGS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <ctime>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/opencl.h>
//#include "C:\Program Files (x86)\AMD APP SDK\3.0\include\CL\opencl.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;
// OpenCL kernel. Each work item takes care of one element of c
char* getKernel(string name){
	string text = "";
	string line;
	ifstream inFile;

	inFile.open(name);
	if (!inFile) {
		cout << "Unable to open file";
		exit(1); // terminate with error
	}

	while (getline(inFile, line)) {
		text += line + "\n";
	}
	inFile.close();
	char *ans = new char[text.length() + 1];
	strcpy(ans, text.c_str());
	return ans;
}

int main(int argc, char* argv[])
{
	
	const int n = 1000000;

	// Device input buffers
	cl_mem d_a;
	// Device output buffer
	cl_mem d_sum;

	cl_platform_id cpPlatform;        // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program
	cl_int err;

	// Allocate memory for each vector on host
	std::vector<float> h_a(n);
	std::vector<float> h_sum(1);

	// Initialize vectors on host
	for (int i = 0; i < n; i++)
		h_a[i] = static_cast<float>(i);

	// print serilize host summation of input vector
	double sum = 0;
	for (int i = 0; i<n; i++)
		sum += h_a[i];
	cout << "host:" << sum << endl;

	// Number of work items in each local work group
	size_t globalSize, localSize;
	localSize = 64;

	// Number of total work items - localSize must be devisor
	globalSize = static_cast<size_t>(ceil(n / (float)localSize)*localSize);

	//calc input and output array sizes
	size_t h_a_bytes = n * sizeof(float);
	size_t h_sum_bytes = static_cast<size_t>(ceil(n / (float)localSize) * sizeof(float));

	// number of loops - the levels of reductions
	size_t loops = static_cast<size_t>(ceil(log(n)/log(localSize)));
	
	// allocate the kernels that target for execution
	cl_kernel *kernels = new cl_kernel[loops];

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	char* kernelStr = getKernel("sum.cl");
	program = clCreateProgramWithSource(context, 1, (const char **)& kernelStr, NULL, &err);

		// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	
	// Create the compute kernels in the program we wish to run
	for (size_t i = 0; i < loops; i++)
		kernels[i] = clCreateKernel(program, "sum", &err);

	// Create the input and output arrays in device memory for our calculation
	d_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY, h_a_bytes, NULL, NULL);
	d_sum = clCreateBuffer(context, CL_MEM_READ_ONLY, h_sum_bytes, NULL, NULL);
	
	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, h_a_bytes, h_a.data(), 0, NULL, NULL);

	// kerenel argument numbering
	bool mode = false;

	size_t length = n;
	// pipeline the kernels into the queue
	for (size_t k = 0; k < loops; mode = !mode, k++) {
		
		// Set the arguments to our compute kernel
		err |= clSetKernelArg(kernels[k], mode, sizeof(cl_mem), &d_a);
		err |= clSetKernelArg(kernels[k], !mode, sizeof(cl_mem), &d_sum);
		err |= clSetKernelArg(kernels[k], 2, sizeof(unsigned int), &length);

		// Execute the kernel over the entire range of the data set  
		err |= clEnqueueNDRangeKernel(queue, kernels[k], 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

		// Redetermine the vectour length after a reduction
		length = static_cast<size_t>(ceil(length / (float)localSize));
		
		//  Redetermine the global work items to lauunch after a reduction
		globalSize = static_cast<size_t>(ceil(length / (float)localSize)*localSize);
	}
	
	// Wait for the command queue to get serviced before reading back results
	err |= clFinish(queue);
	
	// Read the results from the device
	clEnqueueReadBuffer(queue, d_sum, CL_TRUE, 0, sizeof(float), h_sum.data(), 0, NULL, NULL);

	// print the result of the parallel algorithm
	std::cout << "final result: " << h_sum[0] << std::endl;

	// release OpenCL resources
	clReleaseMemObject(d_a);
	clReleaseMemObject(d_sum);
	for (size_t i = 0; i < loops; i++)
		clReleaseKernel(kernels[i]);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	//release host memory
	delete[] kernels;
	delete[] kernelStr;


	system("pause");
	return 0;
}