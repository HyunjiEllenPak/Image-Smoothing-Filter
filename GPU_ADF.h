#ifndef _GPU_ADF_H
#define _GPU_ADF_H
#include <cuda_runtime.h>

extern "C" {
	namespace cglab {
		class ADFilter {
		public:
			/*<input>
			*1.*image:input array pointer, 2.height: image height, 3.width:image width
			*4.lambda:integration constant, 5.kappa:gradient modulus threshold that controls the conduction
			*6.num_iter:iteration number,  7.option:conduction coefficient function proposed by Perona & Malik
			*/
			ADFilter(float *image, int height, int width,  float lambda, float kappa, int iter_num=1, int option=1);
			float* runFilter();
		private:
			float *image = nullptr;//image pixel array
			float *outimage = nullptr;//outimage array to save the computational result
			int height;//image height
			int width;//image weight
			float kappa;
			float lambda;
			int iter_num;
			int option;
		};
	}
}

#endif

