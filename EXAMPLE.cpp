#include<iostream>
#include "gdcm-2.8\gdcmImageReader.h"
#include "gdcm-2.8\gdcmImageWriter.h"
#include"GPU_ADF.h"
#include"GPU_BF.h"
#include<vector>
#include<Windows.h>
#include<string>
/****************************************************************/
//Write these three path
char *input_path="C:/Users\\HJEllenPak\\Documents\\Visual Studio 2015\\Projects\\CUDA_noise_filter\\CUDA_noise_filter\\data\\E0000139\\in_image\\";
char *adf_output_path="C:\\Users\\HJEllenPak\\Documents\\Visual Studio 2015\\Projects\\CUDA_noise_filter\\CUDA_noise_filter\\data\\E0000139\\result_image\\adf\\";
char *bf_output_path="C:\\Users\\HJEllenPak\\Documents\\Visual Studio 2015\\Projects\\CUDA_noise_filter\\CUDA_noise_filter\\data\\E0000139\\result_image\\bf\\";
//Choose a filter kinds and dicom file number which you want to use
#define FILTER_TYPE 0//Execute ADF if 0 or BF 
const int N=10;
//For Anisotropic diffusion filter
float lambda=0.1;//min value is 0 and max value is 0.142857
float kappa=50;
int iter_num=30;
int option=1;
//For bilateral filter
float sigmaD=8;
float sigmaR=0.025;
/***************************************************************/
typedef std::vector<std::string> stringvec;

//char 에서 wchar_t 로의 형변환 함수
wchar_t* ConverCtoWC(const char* str) {
	//wchar_t형 변수 선언
	wchar_t* pStr;
	//멀티 바이트 크기 계산 길이 반환
	int strSize = MultiByteToWideChar(CP_ACP, 0, str, -1, NULL, NULL);
	//wchar_t 메모리 할당
	pStr = new WCHAR[strSize];
	//형 변환
	MultiByteToWideChar(CP_ACP, 0, str, strlen(str) + 1, pStr, strSize);
	return pStr;
}
void read_directory(const char *name, stringvec& v) {
	wchar_t *pattern=ConverCtoWC(name);
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern, &data)) != INVALID_HANDLE_VALUE) {
		do {
			wchar_t* temp=data.cFileName;
			std::wstring wsdata(temp);
			std::string sdata(wsdata.begin(), wsdata.end());
			v.push_back(sdata);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

int main() {
	//Read dicom file
	//Read filenames in the path
	stringvec v;
	std::string input_path_string(input_path);
	input_path_string.append("*.dcm");
	read_directory(input_path_string.c_str(), v);

	char *buf = nullptr;
	float *val = nullptr;
	short *arr=nullptr;
	short *result_arr = nullptr;
	float *result_val=nullptr;// =nullptr;//CPU-memory allocation
	char *result_buf=nullptr;// =nullptr;
	for (int i=0; i < N; i++) {
		//Read DCM file
		const char* infilename=v[i].c_str();
		std::string infile_path(input_path);
		infile_path.append(infilename);
		const char* outfilename=v[i].c_str();
		std::string outfilepath("");
		gdcm::ImageReader reader;		
		reader.SetFileName(infile_path.c_str());
		if (!reader.Read()) {
			std::cerr << "Could not read: " << infilename << std::endl;
			return 1;
		}
		gdcm::Image &inimage = reader.GetImage();
		unsigned buf_length = inimage.GetBufferLength();
		unsigned dimx = inimage.GetDimension(0);
		unsigned dimy = inimage.GetDimension(1);
		unsigned npixels = dimx*dimy;		
		if (i==0) buf = new char[buf_length];		
		inimage.GetBuffer(buf);
		arr = reinterpret_cast<short*>(buf);		
		if (i == 0) val = new float[npixels];

	
		for (int i = 0; i < npixels; i++) {
			val[i] = arr[i];
		}
		//arr[0]=1002.2;
		//for(int i=0;i<npixels;i++)
		//std::cout << "3:"<<arr[0] << std::endl;

#if FILTER_TYPE==0//GPU_adfilter test
		outfilepath=adf_output_path;
		outfilepath.append(outfilename);
		if (i == 0) {
			std::cout << "< Anisotropic Diffusion Filter >" << std::endl;
			std::cout << "lambda: " << lambda << std::endl;
			std::cout << "kappa: " << kappa << std::endl;
			std::cout << "iter_num: " << iter_num << std::endl;
			std::cout << "option: " << option << std::endl;
			std::cout << std::endl;
		}
		cglab::ADFilter gpuADF(val, dimx, dimy, lambda, kappa, iter_num, option);
		result_val=gpuADF.runFilter();
		if (i == 0) result_arr = new short[npixels];
		for (int i = 0; i < npixels; i++)
			result_arr[i] = result_val[i];		
		result_buf = reinterpret_cast<char*>(result_arr);
#elif FILTER_TYPE==1//GPU_bilateral filter test
		outfilepath=bf_output_path;
		outfilepath.append(outfilename);		
		if (i == 0) {
			std::cout << "< Bilateral Filter >" << std::endl;
			std::cout << "sigmaR: " << sigmaR << std::endl;
			std::cout << "sigmaD: " << sigmaD << std::endl;
		}
		cglab::BilateralFilter gpubf(val, dimx, dimy, sigmaD, sigmaR);
		result_val=gpubf.runFilter();		
		if (i == 0) result_arr = new short[npixels];
		for (int i = 0; i < npixels; i++)
			result_arr[i] = result_val[i];
		result_buf = reinterpret_cast<char*>(result_arr);
#endif
		//Write the dicom file
		gdcm::DataElement pixeldata(gdcm::Tag((0x7fe0, 0x0010)));
		pixeldata.SetByteValue((char*)result_buf, buf_length);
		gdcm::DataSet ds;
		ds.Replace(pixeldata);
		inimage.SetDataElement(pixeldata);
		gdcm::ImageWriter writer;
		writer.SetCheckFileMetaInformation(true);
		writer.SetFileName(outfilepath.c_str());
		writer.SetImage(inimage);
		if (!writer.Write()) {
			std::cerr << "Could not write: " << outfilename << std::endl;
			return 1;
		}
	}
	delete[] buf;
	delete[] val;
	//delete[] arr;
	delete[] result_arr;
	//delete[] result_buf;
	system("pause");
	return 0;
}
