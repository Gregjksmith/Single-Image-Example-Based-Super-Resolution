# Single Image Example-based Super-Resolution

## Description

Performs example-based Super-resolution on an image using direct mapping between high and low resolution patches. 

Learning is done utilizing a self-example high-resolution, low-resolution coupled dictionary. Self-example means the dictionary is created from sampling patches from the input low resolution image. This approach is based on the observation that similar patches of a natural image occur within the same scale and between different scales. 

An affine mapping approach is learned to transform a low resolution patch to its high resolution counterpart. This problem is ill-posed and hence we add a regularization term based on Tikhonov regularization.

We take patches of the image in the wavelet domain to perform training and prediction. We expand the discrete wavelet basis by using a discrete rational wavelet transform. Results suggest that the increased sparsity and the larger basis help to improve the ill-posed nature of the problem.

Super-resolution is performed through gradual upscaling. This method better retains low resolution patch structures and outperforms single step upscaling at higher upscaling rates. 


## API

### gs::superResolution

cv::Mat* **gs::superResolution**(cv::Mat inputImage, float upscale, int iterations, double lambda, unsigned int patchSize
							unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight);

> gs::superResolution uses example patches from the input low resolution image to perform super-resolution.

#### Paramaters
**cv::Mat inputImage**: *Input low resolution image.*

**float upscale**: *Upscaling factor. Must be greater than 1.*

**int iterations**: *Number of upscaling iterations performed. The final product is upscaled by a factor of 'upscale'.*

**double lambda**: *Regularization weight.*

**unsigned int patchSize**: *Patch size.*

**unsigned int patchOverlap**: *Patch overlap.*

**unsigned int neighborhoodSize**: *K-nearest neighbors used.*

**double neighborhoodWeight**: *K-nearest neighbor scaling factor. Dictionary atoms are scaled based on the atom's distance from a low resolution patch.*


### gs::superResolutionWavelet

cv::Mat* **gs::superResolutionWavelet**(cv::Mat inputImage, float upscale, int iterations, double lambda, unsigned int patchSize
							unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight, int waveletP, int waveletQ);

> gs::superResolutionWavelet uses example wavelet patches from the input low resolution image to perform super-resolution.

#### Paramaters
**cv::Mat inputImage**: *Input low resolution image.*

**float upscale**: *Upscaling factor. Must be greater than 1.*

**int iterations**: *Number of upscaling iterations performed. The final product is upscaled by a factor of 'upscale'.*

**double lambda**: *Regularization weight.*

**unsigned int patchSize**: *Patch size.*

**unsigned int patchOverlap**: *Patch overlap.*

**unsigned int neighborhoodSize**: *K-nearest neighbors used.*

**double neighborhoodWeight**: *K-nearest neighbor scaling factor. Dictionary atoms are scaled based on the atom's distance from a low resolution patch.*

**int waveletP**: *upsampling wavelet factor.*

**int waveletQ**: *downsampling wavelet factor. WaveletQ must be larger than waveletP.*

## Example

```
#include "SuperResolution.h"
#define LR_IMAGE_PATH "../images/peppers256.tif"
#define GT_IMAGE_PATH "../images/peppers.png"

int main()
{

	cv::Mat imageLR;
	cv::Mat imageGT;

	/*load the test images*/
	cv::Mat imageLR = imread(LR_IMAGE_PATH, CV_LOAD_IMAGE_GRAYSCALE);
	imageLR = gs::to32F(imageLR);
	imageGT = imread(GT_IMAGE_PATH, CV_LOAD_IMAGE_GRAYSCALE);
	imageGT = gs::to32F(imageGT);

	/*SR variables*/
	float upscale = 2.0;
	int iterations = 1;
	unsigned int patchSize = 4;
	unsigned int patchOverlap = 3;
	double lambda = 1e-12;
	unsigned int neighborhoodSize = 200;
	double neighborhoodWeight = 1.0;

	/*Super-resolve the image*/
	Mat* imageSR = gs::superResolve(imageLR, upscale, iterations, lambda, patchSize, patchOverlap, neighborhoodSize, neighborhoodWeight);
	/*export the report and images*/
	gs::exportReport(imageLR, *imageSR, imageGT, upscale, iterations, lambda, patchSize, patchOverlap, neighborhoodSize, neighborhoodWeight);

	int waveletP = 7;
	int waveletQ = 8;
	neighborhoodWeight = 4.0;
	/*Super-resolve the image*/
	Mat* imageSRWavelet = gs::superResolveWavelet(imageLR, upscale, iterations, lambda, patchSize, patchOverlap, neighborhoodSize, neighborhoodWeight, waveletP, waveletQ);
	/*export the report and images*/
	gs::exportReportWavelet(imageLR, *s, imageGT, 2, 1, lambda, patchSize, patchOverlap, neighborhoodSize, neighborhoodWeight,waveletP,waveletQ);
}
```

## Results

###Input Image:

![input image](https://raw.githubusercontent.com/Gregjksmith/Single-Image-Example-Based-Super-Resolution/master/images/peppers256.png?raw=true)

###Super-resolved:

![super-resolved](https://github.com/Gregjksmith/Single-Image-Example-Based-Super-Resolution/blob/master/images/imageSR.png?raw=true)

###Bicubic Interpolation:

![bicubic interpolation](https://raw.githubusercontent.com/Gregjksmith/Single-Image-Example-Based-Super-Resolution/master/images/inputImage_interpolated.png?raw=true)

###Ground Truth:

![ground truth](https://raw.githubusercontent.com/Gregjksmith/Single-Image-Example-Based-Super-Resolution/master/images/imageGT.png?raw=true)

## License

This project is licensed under the terms of the MIT license.

[1] Bevilacqua, Marco, et al. "Single-image super-resolution via linear mapping of interpolated self-examples." IEEE Transactions on Image Processing 23.12 (2014): 5334-5347.