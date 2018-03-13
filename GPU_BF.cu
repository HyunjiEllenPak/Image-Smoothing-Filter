#include <iostream>
#include<algorithm>
#include"GPU_BF.h"
#include "device_launch_parameters.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define M_PI 3.14159265358979323846

using namespace cglab;

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { 
			exit(code); 
			getchar(); 
		}
	}
}
__device__ inline float gaussian(float x, float sigma) {	
	return 1.0f/(sigma*sqrt(2*M_PI))*exp(-(x*x) / (2 * sigma*sigma));
}
__global__ void gpuBFCalculation(float *padimage,float *outimage,float min, float max, float *cGaussian, int pad_height, int pad_width, int kernelRadius, float sigmaD, float sigmaR) {	
	//Calculate our pixel's location
	int x=blockIdx.x*blockDim.x + threadIdx.x;	
	int y=blockIdx.y*blockDim.y + threadIdx.y;
	int out_width=pad_width - kernelRadius * 2;
	int out_height=pad_height - kernelRadius * 2;
	//Boundary check
	if (x < kernelRadius || x >= (pad_height - kernelRadius) || y < kernelRadius || y >= (pad_width - kernelRadius))
		return;

	float sum = 0;
	float totalWeight = 0;
	float centerIntensity = padimage[x*pad_width + y];
	float normCenterIntensity=(centerIntensity - min) / (max - min);
	int kernelSize=kernelRadius * 2 + 1;

	for (int dx=x - kernelRadius; dx <= x + kernelRadius; dx++) {
		for (int dy=y - kernelRadius; dy <= y + kernelRadius; dy++) {
			float kernelPosIntensity=padimage[dx*pad_width + dy];			
			float normKernelPosIntensity=(kernelPosIntensity - min) / (max - min);
			float weight= (cGaussian[dy - y + kernelRadius] * cGaussian[dx - x + kernelRadius]) * gaussian(normKernelPosIntensity - normCenterIntensity, sigmaR);				
			sum+=(weight*kernelPosIntensity);
			totalWeight+=weight;			
		}
	}	
	outimage[(x - kernelRadius) * out_width + (y - kernelRadius)] = sum / totalWeight;
}

BilateralFilter::BilateralFilter(float *image, int height, int width, float sigmaD, float sigmaR) {
	this->image = image;
	this->height = height;
	this->width = width;
	this->sigmaD = sigmaD;
	this->sigmaR = sigmaR;

	outimage=new float[height*width];
	memset(outimage, 0, sizeof(float) * height * width);
	this->kernelRadius= std::log2(std::min(height, width)) / 2;
	//min,max: the values to normalize image intensity
	this->min = image[0];
	this->max = image[0];
	for (int i=0; i < height*width; i++){
		if (image[i] > this->max)
			this->max = image[i];
		if (image[i] < this->min)
			this->min = image[i];
	}	
	//"same" zero-padding	
	this->pad_height=height + kernelRadius * 2;
	this->pad_width=width + kernelRadius * 2;
	padimage=new float[pad_height*pad_width];
	for (int i=0; i < pad_height; i++) {
		for (int j=0; j < pad_width; j++) {
			if (i >= kernelRadius&&i < (pad_height - kernelRadius) && j >= kernelRadius&&j < (pad_width - kernelRadius))
				padimage[i*pad_width + j]=image[(i - kernelRadius)*width + (j - kernelRadius)];
			else
				padimage[i*pad_width + j]=0;
		}
	}
}
float* BilateralFilter::runFilter() {
	//Set gpu device by default
	cudaSetDevice(0);

	float* fGaussian = new float[kernelRadius * 2 + 1];
	float *d_cGaussian;
	float twosigma=sigmaD * 2;
	float sum=0;
	for (int i = 0; i < 2 * kernelRadius + 1; i++){				
		float x=i - kernelRadius;
		fGaussian[i] = expf(-(x*x) / (2 * sigmaD*sigmaD));
		sum+=fGaussian[i];
		
	}
	cudaMalloc(&d_cGaussian, sizeof(float)*(kernelRadius * 2 + 1));
	cudaMemcpy(d_cGaussian, fGaussian, sizeof(float)*(kernelRadius*2 + 1), cudaMemcpyHostToDevice);
	delete[] fGaussian;

	float *d_padimage=nullptr;
	float *d_outimage;
	//Cuda memory allocation and error check
	gpuErrchk(cudaMalloc(&d_padimage, sizeof(float)*pad_width*pad_height));//GPU-memory allocation for d_padimage
	gpuErrchk(cudaMemcpy(d_padimage, padimage, sizeof(float)*pad_width*pad_height, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&d_outimage, sizeof(float)*width*height));
	dim3 threadsPerBlock(16,16);//normally 16*16 is optimal
	dim3 numBlocks(ceil((float)pad_height / threadsPerBlock.x), ceil((float)pad_width / threadsPerBlock.y)); 
	gpuBFCalculation <<<numBlocks, threadsPerBlock >>> (d_padimage,d_outimage,min,max, d_cGaussian, pad_height, pad_width, kernelRadius, sigmaD, sigmaR);
	gpuErrchk(cudaMemcpy(outimage, d_outimage, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
	cudaFree(d_padimage);
	cudaFree(d_outimage);
	cudaFree(d_cGaussian);

	return outimage;

}
