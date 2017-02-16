# Single Image Example-based Super-Resolution

## Description

Performs example-based Super-resolution on an image using direct mapping between high and low resolution patches. High and low resolution patches are taken directly from the input image. A direct mapping trasform is solved by using the k-nearest neighbor patches.

## API

### gs::superResolution

cv::Mat* **gs::superResolution**(cv::Mat& inputImage, float upscale, int iterations, double lambda, unsigned int patchSize
							unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight);

> gs::superResolution uses example patches from the input low resolution image to perform super-resolution.

#### Paramaters
**cv::Mat& inputImage**: *Input low resolution image.*

**float upscale**: *Upscaling factor. Must be greater than 1.*

**int iterations**: *Number of upscaling iterations performed. The final product is upscaled by a factor of 'upscale'.*

**double lambda**: *Regularization weight.*

**unsigned int patchSize**: *Patch size.*

**unsigned int patchOverlap**: *Patch overlap.*

**unsigned int neighborhoodSize**: *K-nearest neighbors used.*

**double neighborhoodWeight**: *K-nearest neighbor scaling factor. Dictionary atoms are scaled based on the atom's distance from a low resolution patch.*


### gs::superResolutionWavelet

cv::Mat* **gs::superResolutionWavelet**(cv::Mat& inputImage, float upscale, int iterations, double lambda, unsigned int patchSize
							unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight, int waveletP, int waveletQ);

> gs::superResolutionWavelet uses example wavelet patches from the input low resolution image to perform super-resolution.

#### Paramaters
**cv::Mat& inputImage**: *Input low resolution image.*

**float upscale**: *Upscaling factor. Must be greater than 1.*

**int iterations**: *Number of upscaling iterations performed. The final product is upscaled by a factor of 'upscale'.*

**double lambda**: *Regularization weight.*

**unsigned int patchSize**: *Patch size.*

**unsigned int patchOverlap**: *Patch overlap.*

**unsigned int neighborhoodSize**: *K-nearest neighbors used.*

**double neighborhoodWeight**: *K-nearest neighbor scaling factor. Dictionary atoms are scaled based on the atom's distance from a low resolution patch.*

**int waveletP**: *upsampling wavelet factor.*

**int waveletQ**: *downsampling wavelet factor. WaveletQ must be larger than waveletP.*