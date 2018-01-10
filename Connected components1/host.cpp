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

int main(int argc, char **argv) {
	const int MAX_PASS = 10;
	char* imStr = "100.png";

	int* value;
	size_t valueSize;
	int Max_Compute_Units;
	cl_platform_id* cpPlatform;       // OpenCL platform
	cl_uint platformCount;			  //number of platforms
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program
	cl_int err = false;

	IplImage *img = 0;
	img = cvLoadImage(imStr, CV_LOAD_IMAGE_COLOR);
	if (!img) {
		cout << "Could not load image " << imStr << endl;
		system("pause");
		exit(1);
	}

	int width = img->width;
	int height = img->height;
	unsigned char *data = (unsigned char *)img->imageData;

	size_t bytes_pixels = width * height * sizeof(cl_int);
	size_t bytes_labels = width * height * sizeof(cl_int);
	size_t bytes_passes = (MAX_PASS + 1) * sizeof(cl_int);

	cl_int *h_pixels = (cl_int *)malloc(bytes_pixels);
	cl_int *h_labels = (cl_int *)malloc(bytes_labels);
	cl_int *h_passes = (cl_int *)malloc(bytes_passes);


	for (int y = 0; y<height; y++) {
		for (int x = 0; x<width; x++) {
			h_pixels[y * width + x] = data[y * img->widthStep + x * 3] > 0 ? 1 : 0;
		}
	}

	// Bind to platform
	clGetPlatformIDs(0, NULL, &platformCount);
	cpPlatform = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
	err |= clGetPlatformIDs(platformCount, cpPlatform, NULL);

	// Get ID for the device
	err |= clGetDeviceIDs(cpPlatform[0], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

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
	program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
	delete[] source;

	err |= clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	cl_kernel kernel_prepare = clCreateKernel(program, "preparation", &err);
	cl_kernel kernel_propagate = clCreateKernel(program, "propagation", &err);

	cl_mem d_pixels = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_pixels, NULL, NULL);
	cl_mem d_labels = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_labels, NULL, NULL);
	cl_mem d_passes = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_passes, NULL, NULL);

	err |= clEnqueueWriteBuffer(queue, d_pixels, CL_TRUE, 0, bytes_pixels, h_pixels, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_labels, CL_TRUE, 0, bytes_labels, h_labels, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_passes, CL_TRUE, 0, bytes_passes, h_passes, 0, NULL, NULL);

	size_t global_size[2] = { (size_t)(width), (size_t)(height) };
	int curpass = 0;

	clSetKernelArg(kernel_prepare, 0, sizeof(cl_mem), (void *)&d_labels);
	clSetKernelArg(kernel_prepare, 1, sizeof(cl_mem), (void *)&d_pixels);
	clSetKernelArg(kernel_prepare, 2, sizeof(cl_mem), (void *)&d_passes);
	clSetKernelArg(kernel_prepare, 3, sizeof(cl_int), (void *)&MAX_PASS);
	clSetKernelArg(kernel_prepare, 4, sizeof(cl_int), (void *)&curpass);
	clSetKernelArg(kernel_prepare, 5, sizeof(cl_int), (int *)&width);
	clSetKernelArg(kernel_prepare, 6, sizeof(cl_int), (int *)&height);

	clEnqueueNDRangeKernel(queue, kernel_prepare, 2, NULL, global_size, NULL, 0, NULL, NULL);

	for (int i = 1; i <= MAX_PASS; i++) {
		clSetKernelArg(kernel_propagate, 0, sizeof(cl_mem), (void *)&d_labels);
		clSetKernelArg(kernel_propagate, 1, sizeof(cl_mem), (void *)&d_pixels);
		clSetKernelArg(kernel_propagate, 2, sizeof(cl_mem), (void *)&d_passes);
		clSetKernelArg(kernel_propagate, 3, sizeof(cl_int), (void *)&MAX_PASS);
		clSetKernelArg(kernel_propagate, 4, sizeof(cl_int), (int *)&width);
		clSetKernelArg(kernel_propagate, 5, sizeof(cl_int), (int *)&height);

		clEnqueueNDRangeKernel(queue, kernel_propagate, 2, NULL, global_size, NULL, 0, NULL, NULL);
	}
	clFinish(queue);

	clEnqueueReadBuffer(queue, d_labels, CL_TRUE, 0, width * height * sizeof(cl_int), h_labels, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, d_passes, CL_TRUE, 0, (MAX_PASS + 1) * sizeof(cl_int), h_passes, 0, NULL, NULL);

	clReleaseMemObject(d_passes);
	clReleaseMemObject(d_labels);
	clReleaseMemObject(d_pixels);
	clReleaseKernel(kernel_propagate);
	clReleaseKernel(kernel_prepare);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	for (int y = 0; y<height; y++) {
		for (int x = 0; x<width; x++) {
			int rgb = h_labels[y * width + x] == -1 ? 0 : (h_labels[y * width + x] * 1103515245 + 12345);
			//int rgb = bufLabel[y * iw + x] == -1 ? 0 : (bufLabel[y * iw + x]);
			data[y * img->widthStep + x * 3 + 0] = rgb & 0xff; rgb >>= 8;
			data[y * img->widthStep + x * 3 + 1] = rgb & 0xff; rgb >>= 8;
			data[y * img->widthStep + x * 3 + 2] = rgb & 0xff; rgb >>= 8;
		}
	}

	int params[3] = { CV_IMWRITE_PNG_COMPRESSION, 9, 0 };

	cvSaveImage("output.png", img, params);

	free(h_passes);
	free(h_labels);
	free(h_pixels);

	free(cpPlatform);

	cvShowImage("color", img);
	cvWaitKey();

	exit(0);
	//time, ccl, random
}