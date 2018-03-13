#include <iostream>
#include<stdio.h>
#include "cuda.h"
#include "device_launch_parameters.h"
#include"GPU_ADF.h"

using namespace cglab;

__global__ void gpuADFCalculation(float *image, float* outimage, int height, int width, float kappa, float lambda, int option) {
	//Calculate our pixel's location
	int x=blockIdx.x*blockDim.x + threadIdx.x;
	int y=blockIdx.y*blockDim.y + threadIdx.y;
	//Boundary check
	if (x<1|| x>=height-1 || y<1 || y>=width-1)
		return;
	float dx = 1.0f;
	float dy = 1.0f;
	float dd = sqrt((float)2);
	float gradN, gradS, gradE, gradW, gradNE, gradNW, gradSE, gradSW = 0;
	float gcN, gcS, gcE, gcW, gcNE, gcNW, gcSE, gcSW;
	float k2 = kappa*kappa;

	gradN = image[(x - 1)*width + y] - image[x*width + y];	
	gradS = image[(x + 1)*width + y] - image[x*width + y];
	gradW= image[x*width + y - 1] - image[x*width + y];
	gradE= image[x*width + y + 1] - image[x*width + y];
	gradNE= image[(x - 1)*width + y + 1] - image[x*width + y];
	gradNW = image[(x - 1)*width + y - 1] - image[x*width + y];
	gradSE = image[(x + 1)*width + y + 1] - image[x*width + y];
	gradSW = image[(x + 1)*width + y - 1] - image[x*width + y];
	//Choose a transfer coefficient function(both are similar)
	if (option == 1) {
		gcN = gradN / (1.0f + gradN*gradN / k2);
		gcS = gradS / (1.0f + gradS*gradS / k2);
		gcW = gradW / (1.0f + gradW*gradW / k2);
		gcE = gradE / (1.0f + gradE*gradE / k2);

		gcNE = gradNE / (1.0f + gradNE*gradNE / k2);
		gcNW = gradNW / (1.0f + gradNW*gradNW / k2);
		gcSE = gradSE / (1.0f + gradSE*gradSE / k2);
		gcSW = gradSW / (1.0f + gradSW*gradSW / k2);
	}
	else if(option==2){
		gcN = gradN*exp(-(gradN*gradN / k2));
		gcS = gradS*exp(-(gradS*gradS / k2));
		gcW = gradW*exp(-(gradW*gradW / k2));
		gcE = gradE*exp(-(gradE*gradE / k2));

		gcNE = gradNE*exp(-(gradNE*gradNE / k2));
		gcNW = gradNW*exp(-(gradNW*gradNW / k2));
		gcSE = gradSE*exp(-(gradSE*gradSE / k2));
		gcSW = gradSW*exp(-(gradSW*gradSW / k2));
	}
	else {
		printf("Option error");
	}
	outimage[x*width +y] = image[x*width + y] + lambda*((1.0f / (dx*dx))*gcN + (1.0f / (dx*dx))*gcS + (1.0f / (dy*dy))*gcW + (1.0f / (dy*dy))*gcE + (1.0f / (dd*dd))*gcNE + (1.0f / (dd*dd))*gcNW + (1.0f / (dd*dd))*gcSE + (1.0f / (dd*dd))*gcSW);
}
ADFilter::ADFilter(float *image, int height, int width,  float lambda, float kappa, int iter_num, int option){
	this->image=image;
	this->iter_num=iter_num;
	this->lambda=lambda;
	this->kappa=kappa;
	this->option=option;
	this->height=height;
	this->width=width;
	outimage = new float[height*width];
	for (int i=0; i < height*width; i++)
		outimage[i]=0;
}
float* ADFilter::runFilter() {
	//Set gpu device by default
	cudaSetDevice(0);
	float *d_image=nullptr;
	float *d_outimage=nullptr;
	//GPU-memory allocation
	cudaMalloc(&d_image, sizeof(float)*width*height);
	cudaMemcpy(d_image, image, sizeof(float)*width*height, cudaMemcpyHostToDevice);
	cudaMalloc(&d_outimage, sizeof(float)*width*height);
	cudaMemset(d_outimage, 0, sizeof(float)*height*width);
	
	dim3 threadsPerBlock(16, 16);//normally 16*16 is optimal
	dim3 numBlocks(height/threadsPerBlock.x, width/threadsPerBlock.y);
	//Run GPU-function by iteration_number times
	for (int i = 0; i < this->iter_num; i++) {
		gpuADFCalculation << <numBlocks, threadsPerBlock >> > (d_image, d_outimage, height, width, kappa, lambda, option);
		if (i < iter_num - 1) {
			cudaMemcpy(d_image, d_outimage, sizeof(float)*width*height, cudaMemcpyDeviceToDevice);
			cudaMemset(d_outimage, 0, sizeof(float)*height*width);
		}
	}		
	cudaMemcpy(outimage, d_outimage, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
	cudaFree(d_image);
	cudaFree(d_outimage);
	return outimage;
}

