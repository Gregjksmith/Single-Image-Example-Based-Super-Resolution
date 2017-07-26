#include "SuperResolution.h"

#define FILTER_SIZE 151
using namespace cv;

gs::RationalWavelet::RationalWavelet(cv::Mat image, int J, int p, int q, int s)
{
	/*create the filter banks*/
	cv::Mat H, G, h, g;
	makeFreqResp(FILTER_SIZE, 1, (float)q / (float)p, s, H, G, h, g);
	this->J = J;
	this->p = p;
	this->q = q;
	this->s = s;

	/*init the wavelet array*/
	wavelet = new Wavelet*[J];
	cv::Mat im1 = image;

	for (int j = 0; j < J; j++)
	{
		wavelet[j] = new Wavelet();

		int N = image.rows;
		int M = image.cols;
		Size imSize = cv::Size(N, M);
		int c = lcm(q, s);
		Size l;
		l.width = c*ceil(imSize.width / c);
		l.height = c*ceil(imSize.height / c);
		cv::Mat xp;
		/*resize the image such that the resulting image can be evenly downsampled*/
		resize(image, xp, l, 0.0, 0.0, CV_INTER_CUBIC);

		cv::Mat lo;
		/*downsample the image im1, and compute the wavelets*/
		afb(&im1, &lo, wavelet[j], &h, &g, p, q, s);

		im1 = lo;
	}

	this->lowband = im1;
}

gs::RationalWavelet::~RationalWavelet()
{
	for (int i = 0; i < J; i++)
	{
		delete wavelet[i];
	}

}

Mat gs::RationalWavelet::inverseRationalWavelet()
{
	/*create the filter banks*/
	cv::Mat H, G, h, g;
	makeFreqResp(FILTER_SIZE, 1, (float)q / (float)p, s, H, G, h, g);

	/* compute the conjugate filters*/
	cv::Mat hconj(1, FILTER_SIZE, CV_32F);
	cv::Mat	gconj(1, FILTER_SIZE, CV_32F);
	for (int i = 0; i < FILTER_SIZE; i++)
	{
		hconj.at<float>(0, i) = h.at<float>(0, FILTER_SIZE - 1 - i);
		gconj.at<float>(0, i) = g.at<float>(0, FILTER_SIZE - 1 - i);
	}

	cv::Mat lo = this->lowband;
	for (int i = J - 1; i >= 0; i--)
	{
		Wavelet* hi = this->wavelet[i];
		/*iterative reconstruct the image*/
		sfb(&lo, hi, &hconj, &gconj, p, q, s);
	}
	return lo;
}

void gs::RationalWavelet::afb(cv::Mat* x, cv::Mat* lo, Wavelet* w, cv::Mat* h, cv::Mat* g, int p, int q, int s)
{
	Size N(ceil(p*x->rows / q), ceil(p*x->cols / q));
	cv::Mat ht;
	transpose(*h, ht);
	cv::Mat gt;
	transpose(*g, gt);

	/*anti alias the input image*/
	filter2D(*x, *lo, -1, *h, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(*lo, *lo, -1, ht, Point(-1, -1), 0.0, BORDER_REPLICATE);
	/*down sample the image*/
	resize(*lo, *lo, N, 0.0, 0.0, CV_INTER_CUBIC);

	/*filter the wavelet LH*/
	filter2D(*x, w->highbandLH, -1, ht, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(w->highbandLH, w->highbandLH, -1, *g, Point(-1, -1), 0.0, BORDER_REPLICATE);

	/*filter the wavelet HL*/
	filter2D(*x, w->highbandHL, -1, gt, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(w->highbandHL, w->highbandHL, -1, *h, Point(-1, -1), 0.0, BORDER_REPLICATE);

	/*filter the wavelet HH*/
	filter2D(*x, w->highbandHH, -1, *g, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(w->highbandHH, w->highbandHH, -1, gt, Point(-1, -1), 0.0, BORDER_REPLICATE);
}

void gs::RationalWavelet::sfb(cv::Mat* lo, Wavelet* hi, cv::Mat* h, cv::Mat* g, int p, int q, int s)
{
	cv::Mat ht;
	transpose(*h, ht);
	cv::Mat gt;
	transpose(*g, gt);
	Size N(hi->highbandHH.rows, hi->highbandHH.cols);
	/*upsacle the image*/
	resize(*lo, *lo, N, 0.0, 0.0, CV_INTER_CUBIC);
	/*anti alias*/
	filter2D(*lo, *lo, -1, *h, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(*lo, *lo, -1, ht, Point(-1, -1), 0.0, BORDER_REPLICATE);

	cv::Mat hiHL;
	cv::Mat hiLH;
	cv::Mat hiHH;

	/*filter the wavelets*/
	filter2D(hi->highbandHL, hiHL, -1, *g, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(hiHL, hiHL, -1, ht, Point(-1, -1), 0.0, BORDER_REPLICATE);

	filter2D(hi->highbandLH, hiLH, -1, *h, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(hiLH, hiLH, -1, gt, Point(-1, -1), 0.0, BORDER_REPLICATE);

	filter2D(hi->highbandHH, hiHH, -1, *g, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(hiHH, hiHH, -1, gt, Point(-1, -1), 0.0, BORDER_REPLICATE);

	//imwrite("../images/testConjFilter0.png", to8U(hiLH));
	//imwrite("../images/testConjFilter1.png", to8U(hiHL));
	//imwrite("../images/testConjFilter2.png", to8U(hiHH));

	/*reconstruct*/
	*lo = *lo + hiHL + hiLH + hiHH;
	//*lo = *lo + hi->highbandHL + hi->highbandLH + hi->highbandHH;
}

int gs::RationalWavelet::gcd(int a, int b)
{
	int c;
	while (a != 0)
	{
		c = a;
		a = b%a;
		b = c;
	}
	return b;
}
int gs::RationalWavelet::lcm(int a, int b)
{
	return a*(b / gcd(a, b));
}

void gs::makeFreqResp(int filterSize, float p, float q, float s, cv::Mat& H, cv::Mat& G, cv::Mat& h, cv::Mat& g)
{
	/*init filters*/
	H = cv::Mat::zeros(cv::Size(1, filterSize), CV_32F);
	G = cv::Mat::zeros(cv::Size(1, filterSize), CV_32F);
	h = cv::Mat::zeros(cv::Size(1, filterSize), CV_32F);
	g = cv::Mat::zeros(cv::Size(1, filterSize), CV_32F);

	float pi = 3.14159265359;
	float wp = ((float)s - 1.0)*pi / s;
	float ws = pi / q;

	float* w;
	int wSize;
	wSize = filterSize;

	/*create sample array, w*/
	w = new float[wSize];
	for (int i = 0; i < wSize; i++)
	{
		w[i] = 2.0*pi*(float)i / float(filterSize);
	}

	/*
	create kPass, pass band magnitude for the low pass filter.
	create kStop, stop band magnitude for the low pass filter.
	create kTrans, stop band magnitude for the low pass filter.
	*/
	float* kPass = new float[wSize];
	float* kStop = new float[wSize];
	float* kTrans = new float[wSize];
	for (int i = 0; i < wSize; i++)
	{
		if (abs(w[i]) <= wp)
			kPass[i] = 1.0;
		else
			kPass[i] = 0.0;

		if (abs(w[i]) >= ws)
			kStop[i] = 1.0;
		else
			kStop[i] = 0.0;

		if (abs(w[i]) > wp && abs(w[i]) < ws)
			kTrans[i] = 1.0;
		else
			kTrans[i] = 0.0;
	}

	float a = (1 - 1.0 / s)*pi;
	float b = (float)p / (float)q - (1.0 - 1.0 / s);
	float* wScaled = new float[wSize];
	for (int i = 0; i < wSize; i++)
	{
		wScaled[i] = (abs(w[i]) - a) / b;
	}


	/*create the low pass magnitude response*/
	for (int i = 0; i < wSize; i++)
	{
		if (kPass[i] == 1.0)
		{
			H.at<float>(0, i) = 1.0;
		}
		else if (kTrans[i] == 1.0)
		{
			float d;
			d = (1.0 + cos(wScaled[i]))*sqrt(2.0 - cos(wScaled[i])) / 2;
			H.at<float>(0, i) = d;
		}
		else
		{
			H.at<float>(0, i) = 0.0;
		}
	}


	if ((filterSize % 2) == 0)
	{
		H.at<float>(0, wSize) = 0.0;
	}

	/*copy and padd the dft with zeros for imag components, set up for the idft*/
	cv::Mat Hpadd = cv::Mat::zeros(cv::Size(1, filterSize), CV_32F);
	int filterMid = floor(filterSize / 2);
	Hpadd.at<float>(0, 0) = H.at<float>(0, 0);
	for (int i = 0; i < filterMid; i++)
	{
		Hpadd.at<float>(0, i * 2 + 1) = H.at<float>(0, i + 1);
	}

	/*
	create kPass, pass band magnitude for the high pass filter.
	create kStop, stop band magnitude for the high pass filter.
	create kTrans, stop band magnitude for the high pass filter.
	*/
	ws = ((float)s - 1.0)*pi / (float)s;
	wp = (float)p*pi / q;
	for (int i = 0; i < wSize; i++)
	{
		if (abs(w[i]) >= wp)
			kPass[i] = 1.0;
		else
			kPass[i] = 0.0;

		if (abs(w[i]) <= ws)
			kStop[i] = 1.0;
		else
			kStop[i] = 0.0;

		if (abs(w[i]) < wp && abs(w[i]) > ws)
			kTrans[i] = 1.0;
		else
			kTrans[i] = 0.0;
	}
	a = (1.0 - 1.0 / s)*pi;
	b = (float)p / q - (1.0 - 1.0 / s);
	for (int i = 0; i < wSize; i++)
	{
		wScaled[i] = (abs(w[i]) - a) / b;
	}

	/*create the high pass magnitude response*/
	for (int i = 0; i < wSize; i++)
	{
		if (kPass[i] == 1.0)
		{
			G.at<float>(0, i) = 1.0;
		}
		else if (kTrans[i] == 1.0)
		{
			float d;
			d = (1.0 - cos(wScaled[i]))*sqrt(2.0 + cos(wScaled[i])) / 2;
			G.at<float>(0, i) = d;
		}
		else
		{
			G.at<float>(0, i) = 0.0;
		}
	}

	if ((filterSize % 2) == 0)
	{
		G.at<float>(0, wSize) = 1.0;
	}

	/*copy and padd the dft with zeros for imag components, set up for the idft*/
	cv::Mat Gpadd = cv::Mat::zeros(cv::Size(1, filterSize), CV_32F);
	Gpadd.at<float>(0, 0) = G.at<float>(0, 0);
	for (int i = 0; i < filterMid; i++)
	{
		Gpadd.at<float>(0, i * 2 + 1) = G.at<float>(0, i + 1);
	}
	Gpadd.at<float>(0, filterSize) = G.at<float>(0, filterMid);
	cv::Mat Hifft;
	cv::Mat Gifft;

	/*take the idft*/
	cv::idft(Hpadd, Hifft, cv::DFT_SCALE);
	cv::idft(Gpadd, Gifft, cv::DFT_SCALE);

	/*shift and center the filters*/
	for (int i = 0; i < filterMid + 1; i++)
	{
		h.at<float>(0, filterMid + i) = Hifft.at<float>(0, i);
		g.at<float>(0, filterMid + i) = Gifft.at<float>(0, i);
	}
	for (int i = 0; i < filterMid; i++)
	{
		h.at<float>(0, i) = Hifft.at<float>(0, i + filterMid + 1);
		g.at<float>(0, i) = Gifft.at<float>(0, i + filterMid + 1);
	}

	/*clean up*/
	delete kPass;
	delete kStop;
	delete kTrans;
	delete w;
	delete wScaled;

	return;
}

cv::Mat* gs::RationalWavelet::waveletBand(int scale, int band)
{
	switch (band)
	{
	case 0:
		return &this->wavelet[scale]->highbandLH;
		break;
	case 1:
		return &this->wavelet[scale]->highbandHL;
		break;
	case 2:
		return &this->wavelet[scale]->highbandHH;
		break;
	}
	return nullptr;

}

cv::Mat gs::RationalWavelet::to32F(cv::Mat image)
{
	Mat copy(image.rows, image.cols, CV_32F);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			copy.at<float>(i, j) = ((float)image.at<unsigned char>(i, j)) / 255.0;
		}
	}
	return copy;
}
cv::Mat gs::RationalWavelet::to8U(cv::Mat image)
{
	Mat copy(image.rows, image.cols, CV_8U);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			int sample = (int)round(abs(image.at<float>(i, j)) * 255.0);
			if (sample > 255)
				sample = 255;
			copy.at<unsigned char>(i, j) = (unsigned char)sample;
		}
	}
	return copy;
}

cv::Mat* gs::superResolveWavelet(cv::Mat inputImage, float upscale, int iterations, double lambda, int patchSize, unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight, int waveletP, int waveletQ)
{
	time_t t = time(0);
	int currentSRIter = 0;

	float gradualUpscale = pow((float)upscale, 1.0 / (float)iterations);
	int waveletJ = ceil(log(1 / gradualUpscale) / log((float)waveletP / (float)waveletQ));
	Mat** imageSR = new Mat*[iterations + 1];

	int numPatches = ceil((inputImage.size().width - patchSize)*(inputImage.size().height - patchSize));

	cv::Mat** dLow;
	cv::Mat** dHigh; 
	cv::flann::Index** tree;

	int waveletScale; 
	int waveletBand;
	int numDictionaries = 3 * waveletJ;
	dLow = new Mat*[numDictionaries];
	dHigh = new Mat*[numDictionaries];
	tree = new cv::flann::Index*[numDictionaries];

	for (int i = 0; i < waveletJ; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int dictionaryIndex = i * 3 + j;
			dLow[dictionaryIndex] = new Mat(3 * numPatches, patchSize*patchSize, CV_32F);
			dHigh[dictionaryIndex] = new Mat(3 * numPatches, patchSize*patchSize, CV_32F);
			createDictionaryWavelet(inputImage, gradualUpscale, patchSize, waveletJ, waveletP, waveletQ, dLow[dictionaryIndex], dHigh[dictionaryIndex], i, j);
			tree[dictionaryIndex] = new cv::flann::Index(*(dLow[dictionaryIndex]), flann::KDTreeIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN);
		}
	}


	imageSR[0] = new Mat(inputImage.rows,inputImage.cols, CV_32F);
	inputImage.copyTo(*(imageSR[0]));
	
	int iter;
	for (iter = 0; iter < iterations; iter++)
	{
		currentSRIter = iter;

		/*compute the SR image size*/
		unsigned int NupX = ceil(imageSR[iter]->size().width * gradualUpscale);
		unsigned int NupY = ceil(imageSR[iter]->size().height * gradualUpscale);
		cv::Size Nup(NupX, NupY);

		Mat imageInterp(NupX, NupY, CV_32F);
		/*resize the input image using bicubic interpolation*/
		resize(*(imageSR[iter]), imageInterp, Nup, 0.0, 0.0, INTER_CUBIC);

		Mat imageFiltered;
		imageFiltered = imageInterp;

		Mat aLowTrans;
		Mat invTemp;
		Mat inv;
		Mat M;
		Mat reconPatch;

		int patchIndex = 0;
		int totalPatches = (floor((float)NupX / (patchSize - patchOverlap)) - 1)*(floor((float)NupY / (patchSize - patchOverlap)) - 1);

		/*init the HR/LR atoms*/
		Mat atomsHigh((unsigned int)(patchSize*patchSize), neighborhoodSize, CV_64F);
		Mat atomsLow((unsigned int)(patchSize*patchSize), neighborhoodSize, CV_64F);

		/*init the patches. We utilize double precision for DM calculations but the final product is single precision*/
		Mat patch(1, patchSize*patchSize, CV_32F);
		Mat patchDouble(patchSize*patchSize, 1, CV_64F);

		/*init the final SR image*/
		Mat* y = new Mat(NupX, NupY, CV_32F);


		/*take the rational wavelet of the filtered image*/
		RationalWavelet* rwUp = new RationalWavelet(imageInterp, waveletJ, waveletP, waveletQ, 1);

		Mat imageFilteredResize;
	
		for (int scale = 0; scale < waveletJ; scale++)
		{
			cv::Size sizeLow(rwUp->waveletBand(scale, 0)->rows, rwUp->waveletBand(scale, 0)->cols);

			/* calculate the variance of the dictionary*/
			for (int band = 0; band < 3; band++)
			{
				Mat wLow = *(rwUp->waveletBand(0, band));
				resize(imageInterp, imageFilteredResize, sizeLow, 0.0, 0.0, CV_INTER_CUBIC);

				int dictIndex = scale * 3 + band;
				patchIndex = 0;

				Mat wImageUp;

				Mat* wSR = rwUp->waveletBand(scale, band);
				Size wSRsize(wSR->rows, wSR->cols);

				resize(imageFilteredResize, wImageUp, wSRsize, 0.0, 0.0, CV_INTER_CUBIC);

#if DEBUG_TEST
				imwrite("../images/waveletInterp.png", to8U(wImageUp));
#endif

				int wX = wImageUp.size().width;
				int wY = wImageUp.size().height;

				Mat recon = Mat::zeros(wX, wY, CV_32F);
				Mat reconGain = Mat::zeros(wX, wY, CV_64F);
				totalPatches = (floor((float)wX / (patchSize - patchOverlap)) - 1)*(floor((float)wY / (patchSize - patchOverlap)) - 1);

				for (int x = 0; x < wX - patchSize + 1; x = x + (patchSize - patchOverlap))
				{
					for (int y = 0; y < wY - patchSize + 1; y = y + (patchSize - patchOverlap))
					{
						double sampleSum = 0.0;
						int pixelIndex = 0;
						for (int i = 0; i < patchSize; i++)
						{
							for (int j = 0; j < patchSize; j++)
							{
								int sampleIndexX = (x + i);
								int sampleIndexY = (y + j);
								float interpSample = (float)wImageUp.at<float>(sampleIndexX, sampleIndexY);
								patch.at<float>(0, pixelIndex) = interpSample;
								patchDouble.at<double>(pixelIndex, 0) = (double)interpSample;
								sampleSum = sampleSum + (double)abs(interpSample);
								pixelIndex++;
							}
						}

						pixelIndex = 0;
						sampleSum = sampleSum / (float)(patchSize*patchSize);
						if (sampleSum != 0.0)
						{
							int pixelIndex = 0;
							for (int i = 0; i < patchSize; i++)
							{
								for (int j = 0; j < patchSize; j++)
								{
									patch.at<float>(0, pixelIndex) = patch.at<float>(0, pixelIndex) - sampleSum;
									patchDouble.at<double>(pixelIndex, 0) = patchDouble.at<double>(pixelIndex, 0) - sampleSum;
									pixelIndex++;
								}
							}
						}

						/* get the k nearest neighbors*/
						nearestNeighborAtoms(&patch, patchSize, tree[dictIndex], dLow[dictIndex], dHigh[dictIndex], neighborhoodSize, neighborhoodWeight, &atomsLow, &atomsHigh);

						/*compute the direct mapping matrix*/
						transpose(atomsLow, aLowTrans);
						invTemp = atomsLow*aLowTrans + lambda*Mat::eye(patchSize*patchSize, patchSize*patchSize, CV_64F);
						inv = invTemp.inv(DECOMP_SVD);
						M = atomsHigh*aLowTrans*inv;
						reconPatch = M*patchDouble;

						pixelIndex = 0;
						for (int i = 0; i < patchSize; i++)
						{
							for (int j = 0; j < patchSize; j++)
							{
								double f = reconPatch.at<double>(pixelIndex, 0);
								recon.at<float>(x + i, y + j) = recon.at<float>(x + i, y + j) + (float)reconPatch.at<double>(pixelIndex, 0);
								reconGain.at<double>(x + i, y + j) = reconGain.at<double>(x + i, y + j) + 1.0;
								pixelIndex++;
							}
						}
						patchIndex++;

						/*print the progress*/
						if (patchIndex % 1000 == 0)
						{
							//system("cls");
							float progress = 100.0*(float)patchIndex / (float)totalPatches;
							printf("Current Iteration: ");
							printf(std::to_string(currentSRIter + 1).c_str());
							printf("/");
							printf(std::to_string(iterations).c_str());
							printf(" scale: ");
							printf(std::to_string(scale + 1).c_str());
							printf("/");
							printf(std::to_string(waveletJ).c_str());
							printf(" band: ");
							printf(std::to_string(band + 1).c_str());
							printf("/3");
							printf(", progress: %0.1f percent \n", progress);
						}
					}
				}

				/*correct the overlapping gain at each pixel*/
				for (int i = 0; i < wX; i++)
				{
					for (int j = 0; j < wY; j++)
					{
						if (reconGain.at<double>(i, j) != 0.0)
						{
							wSR->at<float>(i, j) = (float)(recon.at<float>(i, j) / reconGain.at<double>(i, j));
						}
					}
				}

#if DEBUG_TEST
				imwrite("../images/waveletRecon.png", to8U(*wSR));
#endif

			}
		}
		*y = rwUp->inverseRationalWavelet();

		imageInterp.release();
		delete rwUp;

		imageSR[iter + 1] = y;

		Mat imageSR_ibp;
		/*back project the image for refinement*/
		iterativeBackProjection(*(imageSR[iter + 1]), inputImage, imageSR_ibp, upscale);
		*(imageSR[iter + 1]) = imageSR_ibp;


		std::string iterfp("../images/SR_iteration_");
		iterfp.append(std::to_string(iter));
		iterfp.append(".png");
		imwrite(iterfp.c_str(), to8U(*imageSR[iter + 1]));
	}


	for (int i = 0; i < waveletJ; i++)
	{
		dLow[i]->release();
		dHigh[i]->release();
		delete tree[i];
	}
	delete dLow;
	delete dHigh;
	delete tree;
	
	return imageSR[iter];
}

cv::Mat* gs::superResolve(cv::Mat inputImage, float upscale, int iterations, double lambda, unsigned int patchSize, unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight)
{
	time_t t = time(0);
	float gradualUpscale = pow((float)upscale, 1.0 / (float)iterations);
	cv::Mat** imageSR = new cv::Mat*[iterations + 1];
	cv::Mat trainingImage;
	trainingImage = inputImage;
	unsigned int numPatches = ceil((trainingImage.size().width - patchSize)*(trainingImage.size().height - patchSize));
	int currentSRIter = 0;
	cv::Mat* dLow;
	cv::Mat* dHigh;
	cv::flann::Index* tree;

	dLow = new cv::Mat(numPatches, patchSize*patchSize, CV_32F);
	dHigh = new cv::Mat(numPatches, patchSize*patchSize, CV_32F);
	createDictionary(trainingImage, gradualUpscale, patchSize, dLow, dHigh);
	tree = new cv::flann::Index(*dLow, flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);


	imageSR[0] = new Mat(inputImage.rows, inputImage.cols, CV_32F);
	inputImage.copyTo(*(imageSR[0]));
	int iter;

	for (iter = 0; iter < iterations; iter++)
	{
		currentSRIter = iter;
		/*Super resolve the image*/

		/*compute the SR image size*/
		unsigned int NupX = ceil(imageSR[iter]->size().width * gradualUpscale);
		unsigned int NupY = ceil(imageSR[iter]->size().height * gradualUpscale);
		cv::Size Nup(NupX, NupY);

		cv::Mat imageInterp(NupX, NupY, CV_32F);
		/*resize the input image using bicubic interpolation*/
		resize(*(imageSR[iter]), imageInterp, Nup, 0.0, 0.0, INTER_CUBIC);

		cv::Mat imageFiltered = imageInterp;

		cv::Mat aLowTrans;
		cv::Mat invTemp;
		cv::Mat inv;
		cv::Mat M;
		cv::Mat reconPatch;

		int patchIndex = 0;
		int totalPatches = (floor((float)NupX / (patchSize - patchOverlap)) - 1)*(floor((float)NupY / (patchSize - patchOverlap)) - 1);

		/*init the HR/LR atoms*/
		cv::Mat atomsHigh((unsigned int)(patchSize*patchSize), neighborhoodSize, CV_64F);
		cv::Mat atomsLow((unsigned int)(patchSize*patchSize), neighborhoodSize, CV_64F);

		/*init the patches. We utilize double precision for DM calculations but the final product is single precision*/
		cv::Mat patch(1, patchSize*patchSize, CV_32F);
		cv::Mat patchDouble(patchSize*patchSize, 1, CV_64F);

		/*init the final SR image*/
		cv::Mat* y = new cv::Mat(NupX, NupY, CV_32F);

#if DEBUG_TEST
		imwrite("../images/imageInterpolate.png", to8U(imageFiltered));
#endif

		cv::Mat recon = cv::Mat::zeros(cv::Size(NupX, NupY), CV_64F);
		cv::Mat reconGain = cv::Mat::zeros(cv::Size(NupX, NupY), CV_64F);
		for (int x = 0; x < NupX - patchSize + 1; x = x + (patchSize - patchOverlap))
		{
			for (int y = 0; y < NupY - patchSize + 1; y = y + (patchSize - patchOverlap))
			{
				/* sample patch at (x,y) with size (patchSize,patchSize) */

				int pixelIndex = 0;
				for (int i = 0; i < patchSize; i++)
				{
					for (int j = 0; j < patchSize; j++)
					{
						int sampleIndexX = (x + i);
						int sampleIndexY = (y + j);
						float interpSample = (float)imageFiltered.at<float>(sampleIndexX, sampleIndexY);
						patch.at<float>(0, pixelIndex) = interpSample;
						patchDouble.at<double>(pixelIndex, 0) = (double)interpSample;
						pixelIndex++;
					}
				}

				/*find the k nearest neighbors*/
				nearestNeighborAtoms(&patch, patchSize, tree, dLow, dHigh, neighborhoodSize, neighborhoodWeight, &atomsLow, &atomsHigh);

				/*compute the mapping 'M' between the K-nearest low res atoms and their corresponding K high res atoms*/
				transpose(atomsLow, aLowTrans);
				invTemp = atomsLow*aLowTrans + lambda*cv::Mat::eye(patchSize*patchSize, patchSize*patchSize, CV_64F);
				inv = invTemp.inv(DECOMP_SVD);

				M = atomsHigh*aLowTrans*inv;
				/*estimate the high resolution patch of the input low resolution patch*/
				reconPatch = M*patchDouble;

				pixelIndex = 0;
				for (int i = 0; i < patchSize; i++)
				{
					for (int j = 0; j < patchSize; j++)
					{
						double f = reconPatch.at<double>(pixelIndex, 0);
						/*add the HR estimate patch to the final result*/
						recon.at<double>(x + i, y + j) = recon.at<double>(x + i, y + j) + reconPatch.at<double>(pixelIndex, 0);
						/* compute the scaling factor at each pixel to be divided by at the end*/
						reconGain.at<double>(x + i, y + j) = reconGain.at<double>(x + i, y + j) + 1.0;
						pixelIndex++;
					}
				}
				patchIndex++;

				/*print the progress*/
				if (patchIndex % 500 == 0)
				{
					//system("cls");
					float progress = 100.0*(float)patchIndex / (float)totalPatches;
					printf("Current Iteration: ");
					printf(std::to_string(currentSRIter + 1).c_str());
					printf("/");
					printf(std::to_string(iterations).c_str());
					printf(", progress: %0.1f percent \n", progress);
				}

			}
		}

		for (int i = 0; i < NupX; i++)
		{
			for (int j = 0; j < NupY; j++)
			{
				/*correct the overlapping gain at each pixel*/
				if (reconGain.at<double>(i, j) == 0.0)
				{
					y->at<float>(i, j) = imageInterp.at<float>(i, j);
				}
				else
				{
					y->at<float>(i, j) = (float)(recon.at<double>(i, j) / reconGain.at<double>(i, j));
				}
			}
		}

		imageSR[iter + 1] = y;

		cv::Mat imageSR_ibp;
		/*back project the image for refinement*/
		iterativeBackProjection(*(imageSR[iter + 1]), inputImage, imageSR_ibp, upscale);
		*(imageSR[iter + 1]) = imageSR_ibp;


		std::string iterfp("../images/SR_iteration_");
		iterfp.append(std::to_string(iter));
		iterfp.append(".png");
#if DEBUG_TEST
		imwrite(iterfp.c_str(), to8U(*imageSR[iter + 1]));
#endif

	}

	dLow->release();
	dHigh->release();
	delete tree;

	for (int i = 0; i < iter - 1; i++)
	{
		imageSR[i]->release();
	}

	return imageSR[iter];
}

void gs::createDictionary(cv::Mat inputImage, float gradualUpscale, int patchSize, cv::Mat* dLow, cv::Mat* dHigh)
{
	int nx = (int)ceil(inputImage.size().width);
	int ny = (int)ceil(inputImage.size().height);
	int nx_low = (int)ceil(inputImage.size().width / gradualUpscale);
	int ny_low = (int)ceil(inputImage.size().height / gradualUpscale);

	cv::Mat H, G, h, g;
	/*create the LPF*/
	makeFreqResp(151, 1, (float)gradualUpscale / 1.75, 1, H, G, h, g);
	cv::Mat ht;
	transpose(h, ht);
	cv::Mat imageFiltered;

	/*filter the image, obatin a low res estimate*/
	filter2D(inputImage, imageFiltered, -1, h, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(imageFiltered, imageFiltered, -1, ht, Point(-1, -1), 0.0, BORDER_REPLICATE);

	cv::Mat tt;
	resize(inputImage, tt, Size(nx * 2, ny * 2), 0.0, 0.0, CV_INTER_CUBIC);

	resize(imageFiltered, imageFiltered, Size(nx_low, ny_low), 0.0, 0.0);
	resize(imageFiltered, imageFiltered, Size(nx, ny), 0.0, 0.0, CV_INTER_CUBIC);

#if DEBUG_TEST
	imwrite("../images/trainingImageLR.png", to8U(imageFiltered));
	imwrite("../images/trainingImageHR.png", to8U(inputImage));
	imwrite("../images/trainingImageUpscale.png", to8U(tt));
#endif

	/*iterate through each overlapping patch*/
	int patchIndex = 0;
	for (int i = 0; i < ny - patchSize; i++)
	{
		for (int j = 0; j < nx - patchSize; j++)
		{
			int pixelIndex = 0;
			for (int x = 0; x < patchSize; x++)
			{
				for (int y = 0; y < patchSize; y++)
				{
					int sampleX = (x + i);
					int sampleY = (y + j);
					float sampleLow = (float)imageFiltered.at<float>(sampleX, sampleY);
					float sampleHigh = (float)inputImage.at<float>(sampleX, sampleY);

					/*append to dictionaries*/
					dLow->at<float>(patchIndex, pixelIndex) = sampleLow;
					dHigh->at<float>(patchIndex, pixelIndex) = sampleHigh;
					pixelIndex++;
				}
			}

			patchIndex++;

		}
	}

	imageFiltered.release();


}

void gs::createDictionaryWavelet(cv::Mat inputImage, float gradualUpscale, int patchSize, int waveletJ, int waveletP, int waveletQ, cv::Mat* dLow, cv::Mat* dHigh, int waveletScale, int waveletBand)
{
	int nx = (int)ceil(inputImage.size().width);
	int ny = (int)ceil(inputImage.size().height);
	int nx_low = (int)ceil(inputImage.size().width / gradualUpscale);
	int ny_low = (int)ceil(inputImage.size().height / gradualUpscale);

	cv::Mat H, G, h, g;
	/*create the LPF*/
	gs::makeFreqResp(151, 1, (float)gradualUpscale / 1.75, 1, H, G, h, g);
	cv::Mat ht;
	transpose(h, ht);
	cv::Mat imageFiltered;

	/*filter the image, obatin a low res estimate*/
	filter2D(inputImage, imageFiltered, -1, h, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(imageFiltered, imageFiltered, -1, ht, Point(-1, -1), 0.0, BORDER_REPLICATE);

	cv::Mat tt;
	resize(inputImage, tt, Size(nx * 2, ny * 2), 0.0, 0.0, CV_INTER_CUBIC);

	resize(imageFiltered, imageFiltered, Size(nx_low, ny_low), 0.0, 0.0);
	resize(imageFiltered, imageFiltered, Size(nx, ny), 0.0, 0.0, CV_INTER_CUBIC);

	/*take the rational wavelet transform of the input image*/
	gs::RationalWavelet* rwHR = new gs::RationalWavelet(inputImage, waveletJ, waveletP, waveletQ, 1);

	cv::Mat imageFilteredResize;

	cv::Mat* wImageHR;
	cv::Mat wImageLR;

	int patchIndex = 0;
	//int waveletScale, int waveletBand

	cv::Size sizeLow(rwHR->waveletBand(waveletScale, 0)->rows, rwHR->waveletBand(waveletScale, 0)->cols);
	resize(imageFiltered, imageFilteredResize, sizeLow, 0.0, 0.0, CV_INTER_CUBIC);

	int dictIndex = waveletScale * 3 + waveletBand;
	patchIndex = 0;
	wImageHR = rwHR->waveletBand(waveletScale, waveletBand);
	Size hrSize(wImageHR->rows, wImageHR->cols);

	resize(imageFilteredResize, imageFilteredResize, hrSize, 0.0, 0.0, CV_INTER_CUBIC);

#if DEBUG_TEST
	std::string fpLow("../images/trainingImage_waveletLR_");
	std::string fpHigh("../images/trainingImage_waveletHR_");

	fpLow.append(std::to_string(waveletScale));
	fpLow.append("_");
	fpLow.append(std::to_string(waveletBand));
	fpLow.append(".png");

	fpHigh.append(std::to_string(waveletScale));
	fpHigh.append("_");
	fpHigh.append(std::to_string(waveletBand));
	fpHigh.append(".png");

	//imwrite(fpLow.c_str(), to8U(wImageLR));
	imwrite(fpLow.c_str(), to8U(imageFilteredResize));
	imwrite(fpHigh.c_str(), to8U(*wImageHR));
#endif

	nx = wImageHR->rows;
	ny = wImageHR->cols;

	for (int i = 0; i < nx - patchSize; i++)
	{
		for (int j = 0; j < ny - patchSize; j++)
		{
			int pixelIndex = 0;
			float sumLow = 0.0;
			float sumHigh = 0.0;
			for (int x = 0; x < patchSize; x++)
			{
				for (int y = 0; y < patchSize; y++)
				{
					int sampleX = (x + i);
					int sampleY = (y + j);
					float sampleLow = (float)imageFilteredResize.at<float>(sampleX, sampleY);
					float sampleHigh = (float)wImageHR->at<float>(sampleX, sampleY);
					sumLow = sumLow + (sampleLow);
					sumHigh = sumHigh + abs(sampleHigh);

					/*append the dictionary for the current scale*/
					dLow->at<float>(patchIndex, pixelIndex) = sampleLow;
					dHigh->at<float>(patchIndex, pixelIndex) = sampleHigh;
					pixelIndex++;
				}
			}

			sumLow = sumLow / (float)(patchSize*patchSize);

			pixelIndex = 0;
			if (sumLow != 0.0)
			{
				for (int x = 0; x < patchSize; x++)
				{
					for (int y = 0; y < patchSize; y++)
					{
						//enforce unity dc gain
						dLow->at<float>(patchIndex, pixelIndex) = dLow->at<float>(patchIndex, pixelIndex) - sumLow;
						pixelIndex++;
					}
				}
			}

			/*threshold the patches to avoid too many low enery patches*/
			if (sumHigh > WAVELET_THRESHOLD / ((float)(patchSize*patchSize)))
			{
				patchIndex++;
			}

		}
	}

	cv::Mat dLowTrim(patchIndex, patchSize*patchSize, CV_32F);
	cv::Mat dHighTrim(patchIndex, patchSize*patchSize, CV_32F);
	int pixelIndex = 0;
	/*trim the dictionary to the appropriate size*/
	for (int i = 0; i < patchIndex; i++)
	{
		for (int j = 0; j < patchSize*patchSize; j++)
		{
			dLowTrim.at<float>(i, j) = dLow->at<float>(i, j);
			dHighTrim.at<float>(i, j) = dHigh->at<float>(i, j);
		}
	}

	dLow->release();
	dHigh->release();
	*(dLow) = dLowTrim;
	*(dHigh) = dHighTrim;



	delete rwHR;
}

void gs::nearestNeighborAtoms(cv::Mat* patch, int patchSize, cv::flann::Index* tree, cv::Mat* dLow, cv::Mat* dHigh, int K, double weight, cv::Mat* atomsLow, cv::Mat* atomsHigh)
{
	/*index of the nearest patches patches, sorted by distance*/
	cv::Mat ind;
	/*Eculidean distances of nearest patches*/
	/* NOTE!: DISTANCE IS EUCLIDEAN DISTANCE SQUARED! */
	cv::Mat dist;

	long temp = getTickCount();
	/*search the tree for the K nearest patches of 'patch'*/
	tree->knnSearch(*patch, ind, dist, K);
	long t = getTickCount() - temp;

	double* scaleVec = new double[K];
	for (int i = 0; i < K; i++)
	{
		/* for each K-nearest patches*/
		double d = (double)dist.at<float>(0, i);

		/*compute the atom weighting*/
		double e = -(d) / weight;
		double scale = exp(e);
		scaleVec[i] = scale;
	}

	/*compute the sum of the atoms weights for normalization*/
	double scaleSum = 0.0;
	for (int i = 0; i < K; i++)
	{
		scaleSum = scaleSum + scaleVec[i];
	}
	double tempSum = 0.0;
	for (int i = 0; i < K; i++)
	{
		scaleVec[i] = scaleVec[i] / scaleSum;
		tempSum += scaleVec[i];
	}

	for (int k = 0; k < K; k++)
	{
		int index = ind.at<int>(0, k);
		/*if an error exists, append a zero atom*/
		if (index < dLow->rows && index >= 0 && scaleSum != 0.0)
		{
			/*append the LR atoms to "atomsLow', and HR atoms to 'atomsHigh'. Weight and normalize*/
			for (int j = 0; j < patchSize*patchSize; j++)
			{
				atomsLow->at<double>(j, k) = (double)(dLow->at<float>(index, j)*(scaleVec[k]));
				atomsHigh->at<double>(j, k) = (double)(dHigh->at<float>(index, j)*(scaleVec[k]));
			}
		}
		else
		{
			for (int j = 0; j < patchSize*patchSize; j++)
			{
				atomsLow->at<double>(j, k) = 0.0;
				atomsHigh->at<double>(j, k) = 0.0;
			}
		}
	}

	/*clean up*/
	delete scaleVec;
	ind.release();
	dist.release();
}

#define IBP_ITERS 8
void gs::iterativeBackProjection(cv::Mat imageSR, cv::Mat &imageLR, cv::Mat &result, float upscale)
{
	result = cv::Mat(imageSR.rows, imageSR.cols, CV_32F);
	imageSR.copyTo(result);
	int filterSize = 251;

	/*create the lpf*/
	cv::Mat H, G, h, g;
	makeFreqResp(filterSize, 1.0, upscale , 1.0, H, G, h, g);
	cv::Mat ht;
	transpose(h, ht);
	cv::Mat srFiltered;
	cv::Mat imageErr;

	cv::Mat lr;
	resize(imageLR, lr, Size(imageSR.rows, imageSR.cols), 0.0, 0.0, CV_INTER_CUBIC);
	filter2D(lr, lr, -1, h, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(lr, lr, -1, ht, Point(-1, -1), 0.0, BORDER_REPLICATE);

	for (int i = 0; i < IBP_ITERS; i++)
	{
		/* filter and downsample the sr image */
		filter2D(result, srFiltered, -1, h, Point(-1, -1), 0.0, BORDER_REPLICATE);
		filter2D(srFiltered, srFiltered, -1, ht, Point(-1, -1), 0.0, BORDER_REPLICATE);

		/*compute the error*/
		imageErr = lr - srFiltered;

		/*add the error back*/
		result = result + imageErr;
	}

	/*clean up*/
	H.release();
	G.release();
	h.release();
	g.release();
	ht.release();
	imageErr.release();
	srFiltered.release();
}

cv::Mat gs::to32F(cv::Mat image)
{
	cv::Mat copy(image.rows, image.cols, CV_32F);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			copy.at<float>(i, j) = ((float)image.at<unsigned char>(i, j)) / 255.0;
		}
	}
	return copy;
}

cv::Mat gs::to8U(cv::Mat image)
{
	cv::Mat copy(image.rows, image.cols, CV_8U);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			int sample = (int)round(abs(image.at<float>(i, j)) * 255.0);
			if (sample > 255)
				sample = 255;
			copy.at<unsigned char>(i, j) = (unsigned char)sample;
		}
	}
	return copy;
}

float gs::psnr32F(cv::Mat imageGT, cv::Mat imageRec)
{
	cv::Mat gtResize;
	resize(imageGT, gtResize, imageRec.size(), 0.0, 0.0, CV_INTER_CUBIC);


	double err = 0.0;
	for (int i = 0; i < imageRec.rows; i++)
	{
		for (int j = 0; j < imageRec.cols; j++)
		{
			float recSample = imageRec.at<float>(i, j);
			float gtSample = gtResize.at<float>(i, j);
			float errSample = recSample - gtSample;
			err = err + (double)pow(abs(errSample), 2.0);
		}
	}
	err = err / ((double)imageRec.rows*(double)imageRec.cols);
	double result = 20.0*log10(1.0 / sqrt(err));
	return (float)result;
}

float gs::psnr8U(cv::Mat imageGT, cv::Mat imageRec)
{
	cv::Mat gtResize;
	resize(imageGT, gtResize, imageRec.size(), 0.0, 0.0, CV_INTER_CUBIC);

	unsigned char err = 0;
	for (int i = 0; i < imageRec.rows; i++)
	{
		for (int j = 0; j < imageRec.cols; j++)
		{
			unsigned char recSample = imageRec.at<unsigned char>(i, j);
			unsigned char gtSample = imageGT.at<unsigned char>(i, j);
			unsigned char errSample = recSample - gtSample;
			err = err + (double)pow(abs((double)errSample), 2.0);
		}
	}
	double derr = (double)err / ((double)imageRec.rows*(double)imageRec.cols);
	double result = 20.0*log10(255.0 / sqrt(err));
	return (float)result;
}

void gs::exportReport(Mat inputImage, Mat srImage, double upscale, int iterations, double lambda, unsigned int patchSize, unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight)
{
	time_t t = time(0);
	std::string basePath("../exports/");
	basePath.append(std::to_string((long)t));
	int err = _mkdir(basePath.c_str());

	std::string textPath = basePath;
	textPath.append("/report.txt");

	std::ofstream myfile;
	myfile.open(textPath, std::ofstream::app);

	struct tm * now = localtime(&t);
	myfile << "Date: ";
	myfile << (now->tm_year + 1900);
	myfile << '-';
	myfile << (now->tm_mon + 1);
	myfile << '-';
	myfile << now->tm_mday;
	myfile << "\n";

	myfile << "SR algorithm: Direct Mapping of Self Examples\n";

	myfile << "patch size: ";
	myfile << std::to_string(patchSize);
	myfile << "\n";

	myfile << "patch overlap: ";
	myfile << std::to_string(patchOverlap);
	myfile << "\n";

	myfile << "neighborhood size: ";
	myfile << std::to_string(neighborhoodSize);
	myfile << "\n";

	myfile << "neighborhood weight: ";
	myfile << std::to_string(neighborhoodWeight);
	myfile << "\n";

	myfile << "upscale factor: ";
	myfile << std::to_string(upscale);
	myfile << "\n";

	myfile << "lambda: ";
	myfile << (double)lambda;
	myfile << "\n";

	myfile << "iterations: ";
	myfile << std::to_string(iterations);
	myfile << "\n";

	myfile.close();

	Mat interpImage;
	resize(inputImage, interpImage, srImage.size(), 0.0, 0.0, CV_INTER_CUBIC);
	cv::imwrite(basePath + std::string("/inputImage.png"), gs::to8U(inputImage));
	imwrite(basePath + std::string("/inputImage_interpolated.png"), gs::to8U(interpImage));
	imwrite(basePath + std::string("/imageSR.png"), gs::to8U(srImage));

	basePath.clear();
	textPath.clear();
	interpImage.release();
}

void gs::exportReport(Mat inputImage, Mat srImage, Mat imageGT, double upscale, int iterations, double lambda, unsigned int patchSize, unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight)
{
	time_t t = time(0);
	std::string basePath("../exports/");
	basePath.append(std::to_string((long)t));
	int err = _mkdir(basePath.c_str());

	std::string textPath = basePath;
	textPath.append("/report.txt");

	std::ofstream myfile;
	myfile.open(textPath, std::ofstream::app);

	struct tm * now = localtime(&t);
	myfile << "Date: ";
	myfile << (now->tm_year + 1900);
	myfile << '-';
	myfile << (now->tm_mon + 1);
	myfile << '-';
	myfile << now->tm_mday;
	myfile << "\n";


	myfile << "SR algorithm: Direct Mapping of Self Examples\n";

	myfile << "patch size: ";
	myfile << std::to_string(patchSize);
	myfile << "\n";

	myfile << "patch overlap: ";
	myfile << std::to_string(patchOverlap);
	myfile << "\n";

	myfile << "neighborhood size: ";
	myfile << std::to_string(neighborhoodSize);
	myfile << "\n";

	myfile << "neighborhood weight: ";
	myfile << std::to_string(neighborhoodWeight);
	myfile << "\n";

	myfile << "upscale factor: ";
	myfile << std::to_string(upscale);
	myfile << "\n";

	myfile << "lambda: ";
	myfile << (double)lambda;
	myfile << "\n";

	myfile << "iterations: ";
	myfile << std::to_string(iterations);
	myfile << "\n";


	Mat interpImage;
	resize(inputImage, interpImage, srImage.size(), 0.0, 0.0, CV_INTER_CUBIC);
	cv::imwrite(basePath + std::string("/inputImage.png"), gs::to8U(inputImage));
	imwrite(basePath + std::string("/inputImage_interpolated.png"), gs::to8U(interpImage));
	imwrite(basePath + std::string("/imageSR.png"), gs::to8U(srImage));
	imwrite(basePath + "/imageGT.png", gs::to8U(imageGT));

	Mat imageInter;
	resize(inputImage, imageInter, srImage.size(), 0.0, 0.0, CV_INTER_CUBIC);

	float psnrBC = gs::psnr32F(imageGT, imageInter);
	float psnrSR = gs::psnr32F(imageGT, srImage);

	myfile << "PSNR Super Resoltion: ";
	myfile << std::to_string(psnrSR);
	myfile << "    PSNR Interpolation: ";
	myfile << std::to_string(psnrBC);
	myfile << "    difference: ";
	myfile << std::to_string(psnrSR - psnrBC);
	myfile << "\n";

	myfile.close();

	imageInter.release();
	basePath.clear();
	textPath.clear();
}

void gs::exportReportWavelet(Mat inputImage, Mat srImage, double upscale, int iterations, double lambda, unsigned int patchSize, unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight, int waveletP, int waveletQ)
{
	time_t t = time(0);

	std::string basePath("../exports/");
	basePath.append(std::to_string((long)t));
	int err = _mkdir(basePath.c_str());

	std::string textPath = basePath;
	textPath.append("/report.txt");

	std::ofstream myfile;
	myfile.open(textPath, std::ofstream::app);

	struct tm * now = localtime(&t);
	myfile << "Date: ";
	myfile << (now->tm_year + 1900);
	myfile << '-';
	myfile << (now->tm_mon + 1);
	myfile << '-';
	myfile << now->tm_mday;
	myfile << "\n";

	myfile << "SR algorithm: Direct Mapping of Self Wavelet Examples\n";

	myfile << "patch size: ";
	myfile << std::to_string(patchSize);
	myfile << "\n";

	myfile << "patch overlap: ";
	myfile << std::to_string(patchOverlap);
	myfile << "\n";

	myfile << "neighborhood size: ";
	myfile << std::to_string(neighborhoodSize);
	myfile << "\n";

	myfile << "neighborhood weight: ";
	myfile << std::to_string(neighborhoodWeight);
	myfile << "\n";

	myfile << "upscale factor: ";
	myfile << std::to_string(upscale);
	myfile << "\n";

	myfile << "lambda: ";
	myfile << (double)lambda;
	myfile << "\n";


	myfile << "iterations: ";
	myfile << std::to_string(iterations);
	myfile << "\n";

	myfile << "wavelet dilation factor: ";
	myfile << std::to_string((float)waveletP / (float)waveletQ);
	myfile << "\n";

	myfile.close();

	Mat interpImage;
	resize(inputImage, interpImage, srImage.size(), 0.0, 0.0, CV_INTER_CUBIC);
	cv::imwrite(basePath + std::string("/inputImage.png"), gs::to8U(inputImage));
	imwrite(basePath + std::string("/inputImage_interpolated.png"), gs::to8U(interpImage));
	imwrite(basePath + std::string("/imageSR.png"), gs::to8U(srImage));


	int J = ceil(log(1 / (float)upscale) / log((float)waveletP / (float)waveletQ));
	gs::RationalWavelet* rwSR = new gs::RationalWavelet(srImage, J, waveletP, waveletQ, 1);
	gs::RationalWavelet* rwBC = new gs::RationalWavelet(interpImage, J, waveletP, waveletQ, 1);

	for (int scale = 0; scale < J; scale++)
	{
		for (int band = 0; band < 3; band++)
		{
			Mat* wSR = rwSR->waveletBand(scale, band);
			Mat* wBC = rwBC->waveletBand(scale, band);
			std::string waveletSRpath("/waveletSR_scale");
			waveletSRpath.append(std::to_string(scale));
			waveletSRpath.append("_band");
			waveletSRpath.append(std::to_string(band));
			waveletSRpath.append(".png");
			imwrite(basePath + waveletSRpath, gs::to8U(*wSR));

			std::string waveletBCpath("/waveletInterpolated_scale");
			waveletBCpath.append(std::to_string(scale));
			waveletBCpath.append("_band");
			waveletBCpath.append(std::to_string(band));
			waveletBCpath.append(".png");
			imwrite(basePath + waveletBCpath, gs::to8U(*wBC));
		}
	}

	interpImage.release();
	textPath.clear();
	basePath.clear();
	delete rwSR;
	delete rwBC;
}

void gs::exportReportWavelet(Mat inputImage, Mat srImage, Mat imageGT, double upscale, int iterations, double lambda, unsigned int patchSize, unsigned int patchOverlap, unsigned int neighborhoodSize, double neighborhoodWeight, int waveletP, int waveletQ)
{
	time_t t = time(0);
	
	std::string basePath("../exports/");
	basePath.append(std::to_string((long)t));
	int err = _mkdir(basePath.c_str());

	std::string textPath = basePath;
	textPath.append("/report.txt");

	int J = ceil(log(1 / (float)upscale) / log((float)waveletP / (float)waveletQ));
	gs::RationalWavelet* rwGT = new gs::RationalWavelet(imageGT, J, waveletP, waveletQ, 1);

	for (int scale = 0; scale < J; scale++)
	{
		for (int band = 0; band < 3; band++)
		{
			Mat* wGT = rwGT->waveletBand(scale, band);

			std::string waveletSRpath("/waveletGT_scale");
			waveletSRpath.append(std::to_string(scale));
			waveletSRpath.append("_band");
			waveletSRpath.append(std::to_string(band));
			waveletSRpath.append(".png");
			imwrite(basePath + waveletSRpath, gs::to8U(*wGT));
		}
	}

	imwrite(basePath + "/imageGT.png", gs::to8U(imageGT));
	imwrite(basePath + "/imageSR.png", gs::to8U(srImage));


	std::ofstream myfile;
	myfile.open(textPath, std::ofstream::app);


	struct tm * now = localtime(&t);
	myfile << "Date: ";
	myfile << (now->tm_year + 1900);
	myfile << '-';
	myfile << (now->tm_mon + 1);
	myfile << '-';
	myfile << now->tm_mday;
	myfile << "\n";

	myfile << "SR algorithm: Direct Mapping of Self Wavelet Examples\n";

	myfile << "patch size: ";
	myfile << std::to_string(patchSize);
	myfile << "\n";

	myfile << "patch overlap: ";
	myfile << std::to_string(patchOverlap);
	myfile << "\n";

	myfile << "neighborhood size: ";
	myfile << std::to_string(neighborhoodSize);
	myfile << "\n";

	myfile << "neighborhood weight: ";
	myfile << std::to_string(neighborhoodWeight);
	myfile << "\n";

	myfile << "upscale factor: ";
	myfile << std::to_string(upscale);
	myfile << "\n";

	myfile << "lambda: ";
	myfile << (double)lambda;
	myfile << "\n";


	myfile << "iterations: ";
	myfile << std::to_string(iterations);
	myfile << "\n";

	myfile << "wavelet dilation factor: ";
	myfile << std::to_string((float)waveletP / (float)waveletQ);
	myfile << "\n";

	Mat imageInter;
	resize(inputImage, imageInter, srImage.size(), 0.0, 0.0, CV_INTER_CUBIC);

	float psnrBC = gs::psnr32F(imageGT, imageInter);
	float psnrSR = gs::psnr32F(imageGT, srImage);

	myfile << "PSNR Super Resoltion: ";
	myfile << std::to_string(psnrSR);
	myfile << "    PSNR Interpolation: ";
	myfile << std::to_string(psnrBC);
	myfile << "    difference: ";
	myfile << std::to_string(psnrSR - psnrBC);
	myfile << "\n";

	imwrite(basePath + "/inputImage_interpolated.png", gs::to8U(imageInter));

	gs::RationalWavelet* rwBC = new gs::RationalWavelet(imageInter, J, waveletP, waveletQ, 1);
	gs::RationalWavelet* rwSR = new gs::RationalWavelet(srImage, J, waveletP, waveletQ, 1);

	for (int scale = 0; scale < J; scale++)
	{
		for (int band = 0; band < 3; band++)
		{
			Mat* wBC = rwBC->waveletBand(scale, band);
			Mat* wSR = rwSR->waveletBand(scale, band);
			Mat* wGT = rwGT->waveletBand(scale, band);

			psnrBC = gs::psnr32F(*wGT, *wBC);
			psnrSR = gs::psnr32F(*wGT, *wSR);

			myfile << "PSNR scale ";
			myfile << std::to_string(scale);
			myfile << ", band ";
			myfile << std::to_string(band);
			myfile << ", Super Resolution: ";
			myfile << std::to_string(psnrSR);
			myfile << "   Interpolation: ";
			myfile << std::to_string(psnrBC);
			myfile << "   difference: ";
			myfile << std::to_string(psnrSR - psnrBC);
			myfile << "\n";


			std::string waveletSRpath("/waveletSR_scale");
			waveletSRpath.append(std::to_string(scale));
			waveletSRpath.append("_band");
			waveletSRpath.append(std::to_string(band));
			waveletSRpath.append(".png");
			imwrite(basePath + waveletSRpath, to8U(*wSR));

			std::string waveletBCpath("/waveletInterpolated_scale");
			waveletBCpath.append(std::to_string(scale));
			waveletBCpath.append("_band");
			waveletBCpath.append(std::to_string(band));
			waveletBCpath.append(".png");
			imwrite(basePath + waveletBCpath, to8U(*wBC));
		}
	}

	textPath.clear();
	basePath.clear();
	imageInter.release();

	myfile.close();

	delete rwBC;
	delete rwSR;
	delete rwGT;

}
