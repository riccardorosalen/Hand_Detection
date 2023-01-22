#include "PreProcessing.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/**
 * @brief Segments skin using YCrCb color space taken from DOI?10.35741/issn.0258-2724.55.1.17
 *
 * @param in The image in format cv::Mat using color space YCrCb
 * @return cv::Mat The image mage with the segmentation
 */
cv::Mat skinSeg(cv::Mat img){
	cv::Mat out = img.clone();
	cv::Mat mean_shift;
	cv::pyrMeanShiftFiltering(img, mean_shift, 20, 40);

	cv::Mat in;
	cv::cvtColor(mean_shift, in, cv::COLOR_RGB2YCrCb);

	for (int i = 0; i < in.rows; i++){
		for (int j = 0; j < in.cols; j++){
			//cr
			if (in.at<cv::Vec3b>(i, j)[1] > 137 && in.at<cv::Vec3b>(i, j)[1] < 177){
				out.at<cv::Vec3b>(i, j)[0] = 0;
				out.at<cv::Vec3b>(i, j)[1] = 0;
				out.at<cv::Vec3b>(i, j)[2] = 0;
			}
			//cb
			else if (in.at<cv::Vec3b>(i, j)[2] > 77 && in.at<cv::Vec3b>(i, j)[2] < 127){
				out.at<cv::Vec3b>(i, j)[0] = 0;
				out.at<cv::Vec3b>(i, j)[1] = 0;
				out.at<cv::Vec3b>(i, j)[2] = 0;
			}
			else if (in.at<cv::Vec3b>(i, j)[2] + in.at<cv::Vec3b>(i, j)[1] * 0.6 > 190 && in.at<cv::Vec3b>(i, j)[2] + in.at<cv::Vec3b>(i, j)[1] * 0.6 < 215){
				out.at<cv::Vec3b>(i, j)[0] = 0;
				out.at<cv::Vec3b>(i, j)[1] = 0;
				out.at<cv::Vec3b>(i, j)[2] = 0;
			}
			else if (in.at<cv::Vec3b>(i, j)[2] > 105 && in.at<cv::Vec3b>(i, j)[2] < 140) {
				if (in.at<cv::Vec3b>(i, j)[1] > 140 && in.at<cv::Vec3b>(i, j)[1] < 160) {
					out.at<cv::Vec3b>(i, j)[0] = 0;
					out.at<cv::Vec3b>(i, j)[1] = 0;
					out.at<cv::Vec3b>(i, j)[2] = 0;
				}
			}
		}
	}
	return out;
}

/**
 * @brief Apply closing operation to the image
 *
 * @param in The image to be processed
 * @return cv::Mat The image with the morphological operations applied
 */
cv::Mat morph(cv::Mat img){
	cv::Mat dst;
	int operation = 2;
	int morph_size = 8;
	int morph_elem = 2; //0 rect, 1 cross, 2 ellipse
	//Not always the image is grayscale
	if (img.channels() > 1) {
		cv::cvtColor(img, img, cv::COLOR_RGB2GRAY, 0);
	}
	cv::Mat element = getStructuringElement(morph_elem, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
	cv::morphologyEx(img, dst, operation, element);
	return dst;
}


/**
 * @brief Apply several closing operation to the image
 *
 * @param in The image to be processed
 * @return cv::Mat A mask containing all the pixel left coloured from operations performed
 */
cv::Mat morphMask(cv::Mat img) {
	cv::Mat opened_image = (morph(morph(morph(morph(skinSeg(img))))));
	cv::Mat	mask;
	img.copyTo(mask);
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (opened_image.at<uint8_t>(i, j) == 0) {
				mask.at<cv::Vec3b>(i, j) = (0, 0, 0);
			}
		}
	}
	return mask;
}