/*
Greg Smith 2016.

Single image example-based Super-resolution using direct mapping of self-examples. based on :
Bevilacqua, Marco, et al. "Single-image super-resolution via linear mapping of interpolated self-examples." IEEE Transactions on Image Processing 23.12 (2014): 5334-5347.

We create a two dictionaries of self example wavelet patches: a high resolution dictionary and a low resolution dictionary.
The high resolution patches are sampled directly from the input image and the low resolution patches are sample from the input 
image passed through a low pass filter. We first upscale the image using bicubic interpolation. For each wavelet patch, he K-nearest 
neighbor atoms are searched from the low resolution dictionary. The corresponding high resolution dictionary atoms are 
combined to estimate a high resolution patch. The patches for each wavelet band and scale are combined to form the super-resolved 
image.

Gradual upscaling is performed to improve the SR results.

Hyperparameters:

float. Upscale. Upscaling factor. 
int. Iterations. Number of iterations performed to achieve the desired upscaling.
int. Patch Size. Dimension in pixels of the patch size. 
int. Patch Overlap. Dimension in pixels of the patch overlap.
float. Lambda. Regularization weight.
int. Neighborhood Size. Number of nearest neighbors used in K-nearest neighbor search. 
int. Neighborhood Weight. Atom exponential weighting variance. 

Wavelet only hyperparameters:
int. P. wavelet dilation factor p. 
int. Q. wavelet dilation factor q. 

*/

#pragma once

#define DEBUG_TEST 1
#define PI (double)3.14159265359
#define WAVELET_THRESHOLD 0.001

#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <time.h>
#include <stdio.h>
#include <fstream>
#include <stdlib.h> 
#include <direct.h>  

namespace gs
{
	/*Rational Wavelet
	Implements a wavelet transform and inverse transform with a rational dilation factor.
	author: gsmith
	*/

	/*
	Wavelet struct
	@members, variables:
	Mat highbandLH, wavelet band, horizontal low frequency, verical high frequency.
	Mat highbandHL, wavelet band, horizontal high frequency, verical low frequency.
	Mat highbandHH, wavelet band, horizontal high frequency, verical high frequency.
	*/
	struct Wavelet
	{
		cv::Mat highbandLH;
		cv::Mat highbandHL;
		cv::Mat highbandHH;
	};

	/*
	MakeFreqResp
	creates two wavelet filters h and g. h is a lowpass filter and g is a high pass filter.

	@param int filtersize, N
	@param float p, Wavelet upscale
	@param float q, Wavelet downscale
	@param float s, Wavelet band downsample fator, normally set to 1.
	@param Mat& H, Magnitude of the dft of filter h.
	@param Mat& G, Magnitude of the dft of filter g.
	@param Mat& h, low pass filter
	@param Mat& g, high pass filter.
	*/
	void makeFreqResp(int filterSize, float p, float q, float s, cv::Mat& H, cv::Mat& G, cv::Mat& h, cv::Mat& g);

	/*
	class RationalWavelet
	performes the rational wavelet wavelet of an image using the number of scales J, dilation factors p,q and s.

	@members,
	functions:
	RationalWavelet, constructor, overload 0.
	~RationalWavelet(), destructor.
	Mat inverseRationalWavlet(), performs the inverse rational wavelet using Wavelet* wavelet and lowband.
	Mat* waveletBand(int scale, int band), returns a pointer to the wavelet band at a specified scale and band

	variables:
	Mat lowband, baseband, lowresolution image.
	*/
	class RationalWavelet
	{
	public:

		/*constructor
		@param Mat image, input image to be decomposed
		@param int J, number of scales
		@param int p, upscale factor
		@param int q, downscale factor
		@param int s, wavelet downsampling factor
		*/
		RationalWavelet(cv::Mat image, int J, int p, int q, int s);

		/*destructor*/
		~RationalWavelet();

		/*inverseRationalWavelet
		@return Mat, returns the inverse rational wavelet of Wavelet** wavelet
		*/
		cv::Mat inverseRationalWavelet();

		/*
		waveletBand
		@param int scale,
		@oaram int band
		@return Mat*, returns a pointer to the wavelet band at the specificed scale and band.
		*/
		cv::Mat* waveletBand(int scale, int band);

		/*baseband low resolution image*/
		cv::Mat lowband;
	private:
		/*scale and dilation factors*/
		int J, p, q, s;

		/*
		analysis filter bank
		performs the radwt. Filters and downsamples image x.

		@param Mat* x, input image.
		@param Mat* lo, resulting baseband low resolution image.
		@param Wavelet* w, resulting wavelet bands.
		@param Mat* h, low pass filter bank.
		@param Mat* g, high pass filter bank.
		@param int p, upscale factor.
		@param int q, downscale factor.
		@param int s, wavelet downsampling factor.
		*/
		void afb(cv::Mat* x, cv::Mat* lo, Wavelet* w, cv::Mat* h, cv::Mat* g, int p, int q, int s);

		/*
		synthesis filter bank
		Reconstructs an image using wavelet bands an a downsampled image.

		@param Mat* lo, input low resolution base band image. Reconstructed image is returned in lo.
		@param Wavelet* hi, input wavelet bands.
		@param Mat* h, low pass filter bank.
		@param Mat* g, high pass filter bank.
		@param int p, upscale factor.
		@param int q, downscale factor.
		@param int s, wavelet downsampling factor.
		*/
		void sfb(cv::Mat* lo, Wavelet* hi, cv::Mat* h, cv::Mat* g, int p, int q, int s);

		/*array of wavelet bands*/
		Wavelet** wavelet;

		/*converts an unsigned char image into a floating point image.*/
		cv::Mat to32F(cv::Mat im);

		/*converts a floating point image into an unsigned char image*/
		cv::Mat to8U(cv::Mat im);

		/*computes the greatest common denominator between int a and int b*/
		int gcd(int a, int b);

		/*computes the lowest common denominator between int a and int b*/
		int lcm(int a, int b);
	};


	/*
	superResolve

	Takes the input image and applied the super resolution algorithm.

	@param Mat. inputImage.
	@param float. upscale.
	@param double. lambda.
	@param unsigned int. patchSize.
	@param unsigned int. patchOverlap.
	@param unsigned int. neighborhoodSize.
	@param double. neighborhoodWeight.

	@return Mat*
	*/
	cv::Mat* superResolve(cv::Mat inputImage, float upscale, int iterations, double lambda, unsigned int patchSize, unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight);



	/*
	superResolve

	Takes the input image and applied the super resolution algorithm. uses a wavelet reconstruction approach.

	@param Mat. inputImage.
	@param float. upscale.
	@param double. lambda.
	@param unsigned int. patchSize.
	@param unsigned int. patchOverlap.
	@param unsigned int. neighborhoodSize.
	@param double. neighborhoodWeight.

	@return Mat*
	*/
	cv::Mat* superResolveWavelet(cv::Mat inputImage, float upscale, int iterations, double lambda, int patchSize, unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight, int waveletP, int waveletQ);



	/*
	nearestNeighborAtoms

	nearestNeighborAtoms takes an input patch and searches the low resolution dictionary for the K nearest neighbors. The found low resolution and corresponding high resolution
	atoms are weighted exponentially with a variance of 'weight'.

	@param Mat*. patch. The input path to be searched.
	@param cv::flann::Index*. tree. Binary tree built from the low resolution dictionary used for efficient searching of the dictionary
	@param Mat*. dLow. Low resolution dictionary.
	@param Mat*. dHigh. high resolution dictionary.
	@param int. K. Number of nearest neighbors to be searched.
	@param double. weight. Exponential weighting variance.
	@param Mat*. atomsLow. Resulting searched and weighted low resolution atoms.
	@param Mat*. atomsHigh. Resulting searched and weighted high resolution atoms.

	@return void
	*/
	void nearestNeighborAtoms(cv::Mat* patch, int patchSize, cv::flann::Index* tree, cv::Mat* dLow, cv::Mat* dHigh, int K, double weight, cv::Mat* atomsLow, cv::Mat* atomsHigh);


	/*
	createDictionary
	creates a high/low resolution coupled dictionary from an image (inputImage). With the SR Wavelet build configs, a wavelet patch dictionary is created.
	With wavelets an array of dictionaries is created, one for each scale of the wavelet.

	@param cv::Mat. inputImage
	@param float. upscale

	@param Mat*. dLow. Low resolution dictionary (SR full band)
	@param Mat*. dHigh. High resolution dictionary (SR full band)

	@param Mat**. dLow. Low resolution dictionary (SR wavelet)
	@param Mat**. dHigh. High resolution dictionary (SR wavelet)

	@return void
	*/
	void createDictionary(cv::Mat inputImage, float gradualUpscale, int patchSize, cv::Mat* dLow, cv::Mat* dHigh);

	void createDictionaryWavelet(cv::Mat inputImage, float gradualUpscale, int patchSize, int waveletJ, int waveletP, int waveletQ, cv::Mat* dLow, cv::Mat* dHigh, int waveletScale, int waveletBand);


	/*
	iterativeBackProjection

	Iterative back projection computes the error between the input image and the low resolution super resolution estimate. We back project the
	error back onto the estime for a better estimate. We add an early stopping criteria to avoid gibbs phenomena.

	@param Mat. imageSR. Super resolution estimate.
	@param &Mat. imageLR. Input low resolution image.
	@param &Mat. result. Backprojected super resolution image.
	@param float. upscale.

	@return void
	*/
	void iterativeBackProjection(cv::Mat imageSR, cv::Mat &imageLR, cv::Mat &result, float upscale);



	/* to32F
	converts an unsigned char image to a floating point image.

	@param Mat image. an unsigned char image.

	@return Mat, a floating point image.
	*/
	cv::Mat to32F(cv::Mat image);


	/* to8U
	converts an floating point image to an unsigned char image.

	@param Mat image. a floating point image.

	@return Mat, an unsigned char image.
	*/
	cv::Mat to8U(cv::Mat image);


	/* psnr32F
	computes the peak signal to noise ration between two images: imageGT, imageRec.
	both images are asumed to be floating point images.

	@param Mat imageGT.
	@param Mat imageRec.

	@return flaot, psnr
	*/
	float psnr32F(cv::Mat imageGT, cv::Mat imageRec);


	/* psnr8U
	computes the peak signal to noise ration between two images: imageGT, imageRec.
	both images are asumed to unsigned char images.

	@param Mat imageGT.
	@param Mat imageRec.

	@return flaot, psnr
	*/
	float psnr8U(cv::Mat imageGT, cv::Mat imageRec);
	

	void exportReport(cv::Mat inputImage, cv::Mat srImage, double upscale, int iterations, double lambda, unsigned int patchSize, unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight);
	void exportReport(cv::Mat inputImage, cv::Mat srImage, cv::Mat imageGT, double upscale, int iterations, double lambda, unsigned int patchSize, unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight);

	void exportReportWavelet(cv::Mat inputImage, cv::Mat srImage, double upscale, int iterations, double lambda, unsigned int patchSize, unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight, int waveletP, int waveletQ);
	void exportReportWavelet(cv::Mat inputImage, cv::Mat srImage, cv::Mat imageGT, double upscale, int iterations, double lambda, unsigned int patchSize, unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight, int waveletP, int waveletQ);
}