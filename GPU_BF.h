#ifndef _GPU_BF_H
#define _GPU_BF_H
#include <cuda_runtime.h>

extern "C" {
	namespace cglab {
		class BilateralFilter {
		public:
			/*<input>
			*1.image: input image 2.outimage: output image 3.height: image height 4.width: image width
			5.sigmaD: domain filter gaussian sigma 6.sigmaR: range filter gaussian sigma(based on intensity similarity)
			*/
			BilateralFilter(float *image, int height, int width, float sigmaD, float sigmaR);
			float* runFilter();
		private:
			float *image = nullptr;//image pixel array
			float *outimage = nullptr;//temporary output-image array to save the computational result
			int height;//image height
			int width;//image width
			int pad_height;//padding image height
			int pad_width;//padding image width
			int kernelRadius;
			float sigmaR;
			float sigmaD;				
			float *padimage=nullptr;
			float min;
			float max;
		};
	}
}
#endif