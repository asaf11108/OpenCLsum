#define CL_ENABLE_EXCEPTIONS
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <CL/opencl.h>

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

char strbuf[10010] = "\0";


#define MAXPLATFORMS 10
#define MAXDEVICES 10

using namespace std;
// OpenCL kernel. Each work item takes care of one element of c
char* getKernel(string name) {
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



void openclErrorCallback(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
	fprintf(stderr, "\nError callback called, info = %s\n", errinfo);
}

#define MAXPASS 10

int main(int argc, char **argv) {
	int i;
	char* imStr = "test.png";

	int* value;
	size_t valueSize;
	int Max_Compute_Units;
	cl_platform_id* cpPlatform;       // OpenCL platform
	cl_uint platformCount;			  //number of platforms
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program_red;               // program
	cl_program program_ser;               // program
	cl_int err;

	IplImage *img = 0;
	img = cvLoadImage(imStr, CV_LOAD_IMAGE_COLOR);
	if (!img) {
		cout << "Could not load image " << imStr << endl;
		system("pause");
		exit(1);
	}

	if (img->nChannels != 3) {
		cout << "nChannels != 3\n" << endl;
		system("pause");
		exit(1);
	}
	int width = img->width;
	int height = img->height;
	unsigned char *data = (unsigned char *)img->imageData;

	//

	cl_int *bufPix = (cl_int *)malloc(width * height * sizeof(cl_int));
	cl_int *bufLabel = (cl_int *)malloc(width * height * sizeof(cl_int));
	cl_int *bufFlags = (cl_int *)malloc((MAXPASS + 1)* sizeof(cl_int));


	for (int y = 0; y<height; y++) {
		for (int x = 0; x<width; x++) {
			bufPix[y * width + x] = data[y * img->widthStep + x * 3] > 0 ? 1 : 0;
		}
	}

	// Bind to platform
	clGetPlatformIDs(0, NULL, &platformCount);
	cpPlatform = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
	err = clGetPlatformIDs(platformCount, cpPlatform, NULL);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform[0], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	// get devices info
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, 0, NULL, &valueSize);
	value = (int*)malloc(valueSize);
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, valueSize, value, NULL);
	Max_Compute_Units = *value;
	free(value);
	std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS:" << Max_Compute_Units << endl;

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	queue = clCreateCommandQueue(context, device_id, 0, &err);

	char *source = getKernel("ccl.cl");
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
	delete[] source;

	err |= clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	cl_kernel kernel_prepare = clCreateKernel(program, "labelxPreprocess_int_int", &err);
	cl_kernel kernel_propagate = clCreateKernel(program, "label8xMain_int_int", &err);

	// By specifying CL_MEM_COPY_HOST_PTR, device buffers are cleared.
	cl_mem memPix = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, width * height * sizeof(cl_int), bufPix, NULL);
	cl_mem memLabel = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, width * height * sizeof(cl_int), bufLabel, NULL);
	cl_mem memFlags = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (MAXPASS + 1) * sizeof(cl_int), bufFlags, NULL);

	size_t work_size[2] = { (size_t)((width + 31) & ~31), (size_t)((height + 31) & ~31) };

	cl_event events[MAXPASS + 1];
	for (i = 0; i <= MAXPASS; i++) {
		events[i] = clCreateUserEvent(context, NULL);
	}

	//

	clSetKernelArg(kernel_prepare, 0, sizeof(cl_mem), (void *)&memLabel);
	clSetKernelArg(kernel_prepare, 1, sizeof(cl_mem), (void *)&memPix);
	clSetKernelArg(kernel_prepare, 2, sizeof(cl_mem), (void *)&memFlags);
	i = MAXPASS; clSetKernelArg(kernel_prepare, 3, sizeof(cl_int), (void *)&i);
	i = 0; clSetKernelArg(kernel_prepare, 4, sizeof(cl_int), (void *)&i);
	clSetKernelArg(kernel_prepare, 5, sizeof(cl_int), (int *)&width);
	clSetKernelArg(kernel_prepare, 6, sizeof(cl_int), (int *)&height);

	clEnqueueNDRangeKernel(queue, kernel_prepare, 2, NULL, work_size, NULL, 0, NULL, &events[0]);

	for (i = 1; i <= MAXPASS; i++) {
		clSetKernelArg(kernel_propagate, 0, sizeof(cl_mem), (void *)&memLabel);
		clSetKernelArg(kernel_propagate, 1, sizeof(cl_mem), (void *)&memPix);
		clSetKernelArg(kernel_propagate, 2, sizeof(cl_mem), (void *)&memFlags);
		clSetKernelArg(kernel_propagate, 3, sizeof(cl_int), (void *)&i);
		clSetKernelArg(kernel_propagate, 4, sizeof(cl_int), (int *)&width);
		clSetKernelArg(kernel_propagate, 5, sizeof(cl_int), (int *)&height);

		clEnqueueNDRangeKernel(queue, kernel_propagate, 2, NULL, work_size, NULL, 0, NULL, &events[i]);
	}

	clEnqueueReadBuffer(queue, memLabel, CL_TRUE, 0, width * height * sizeof(cl_int), bufLabel, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, memFlags, CL_TRUE, 0, (MAXPASS + 1) * sizeof(cl_int), bufFlags, 0, NULL, NULL);

	clFinish(queue);

	long long int total = 0;
	for (i = 0; i <= MAXPASS; i++) {
		cl_ulong tstart, tend;
		clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tstart, NULL);
		clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL);
		clReleaseEvent(events[i]);

		printf("pass %2d : %10lld nano sec\n", i, (long long int)(tend - tstart));
		total += tend - tstart;
	}

	printf("total   : %10lld nano sec\n", total);

	clReleaseMemObject(memFlags);
	clReleaseMemObject(memLabel);
	clReleaseMemObject(memPix);
	clReleaseKernel(kernel_propagate);
	clReleaseKernel(kernel_prepare);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	//

	for (int y = 0; y<height; y++) {
		for (int x = 0; x<width; x++) {
			int rgb = bufLabel[y * width + x] == -1 ? 0 : (bufLabel[y * width + x] * 1103515245 + 12345);
			//int rgb = bufLabel[y * iw + x] == -1 ? 0 : (bufLabel[y * iw + x]);
			data[y * img->widthStep + x * 3 + 0] = rgb & 0xff; rgb >>= 8;
			data[y * img->widthStep + x * 3 + 1] = rgb & 0xff; rgb >>= 8;
			data[y * img->widthStep + x * 3 + 2] = rgb & 0xff; rgb >>= 8;
		}
	}

	int params[3] = { CV_IMWRITE_PNG_COMPRESSION, 9, 0 };

	cvSaveImage("output.png", img, params);

	free(bufFlags);
	free(bufLabel);
	free(bufPix);

	free(cpPlatform);

	cvShowImage("color", img);
	cvWaitKey();

	exit(0);
}