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
	
	const int n = 1000;

	// Host input vectors
	// Host output vector
	std::vector<float> h_sum(1);

	// Device input buffers
	cl_mem d_a;
	// Device output buffer
	cl_mem d_sum;

	cl_platform_id cpPlatform;        // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program
	cl_kernel kernel;                 // kernel

	// Allocate memory for each vector on host
	std::vector<float> h_a(n);

	// Initialize vectors on host
	int i;
	for (i = 0; i < n; i++)
	{
		h_a[i] = static_cast<float>(i);
	}

	size_t globalSize, localSize;
	cl_int err;

	// Number of work items in each local work group
	localSize = 64;

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize)*localSize;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source buffer
	char* ans = getKernel("sum.cl");
	program = clCreateProgramWithSource(context, 1,
		(const char **) & ans, NULL, &err);
	delete[] ans;

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "sum", &err);

	// Create the input and output arrays in device memory for our calculation
	d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, h_a.size()*sizeof(float), NULL, NULL);
	d_sum = clCreateBuffer(context, CL_MEM_WRITE_ONLY, h_sum.size()*sizeof(float), NULL, NULL);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, h_a.size()*sizeof(float), h_a.data(), 0, NULL, NULL);

	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_sum);
	err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &n);

	const clock_t begin_time = clock();
	// Execute the kernel over the entire range of the data set  
	err |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	err |= clFinish(queue);
	cout << "diff: " << float(clock() - begin_time) / CLOCKS_PER_SEC << endl;
	// Read the results from the device
	clEnqueueReadBuffer(queue, d_sum, CL_TRUE, 0,
		h_sum.size()*sizeof(float), h_sum.data(), 0, NULL, NULL);

	//Sum up vector c and print result divided by n, this should equal 1 within error
	std::cout << "final result: " << h_sum[0] << std::endl;

	// release OpenCL resources
	clReleaseMemObject(d_a);
	clReleaseMemObject(d_sum);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	//release host memory

	double sum = 0;
	for (int i = 0; i<n; i++){
		sum += h_a[i];
	}

	cout << "host:" << sum << endl;
	
	system("pause");
	//
	return 0;
}