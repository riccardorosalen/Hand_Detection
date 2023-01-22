#include <iostream>

#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "ImgOps.h"
#include "Utility.h"
#include "PreProcessing.h"
#include <filesystem>

 /**
  * @brief Extract keypoints and associated descriptors with SIFT method and create an image with key points
  *
  * @param image the input image
  * @param outKeyPoints the output vector containing a list of key points
  * 
  * @return cv::Mat the descriptors
  */
cv::Mat extractKeypoints(cv::Mat const& image, std::vector<cv::KeyPoint>& outKeyPoints) {
	cv::Mat outDescriptors;
	cv::Ptr<cv::SIFT> f2d = cv::SIFT::create();
	f2d->detect(image, outKeyPoints);
	f2d->compute(image, outKeyPoints, outDescriptors);	
	return outDescriptors;
}

/**
 * @brief Extract keypoints and associated descriptors with SIFT method and create an image with key points
 *
 * @param image the input image
 * @param outKeyPoints the output vector containing a list of key points
 * @param outDescriptors the output cv::Mat containing the the key points descriptors
 * @param outImage the output image with keypoints drawn in it
 */
cv::Mat extractKeypoints(cv::Mat const &image, std::vector<cv::KeyPoint> &outKeyPoints, cv::Mat &outDescriptors){
	cv::Mat out;
	cv::Ptr<cv::SIFT> f2d = cv::SIFT::create();

	f2d->detect(image, outKeyPoints);
	f2d->compute(image, outKeyPoints, outDescriptors);
	if(outKeyPoints.size()>0){
		cv::drawKeypoints(image, outKeyPoints, out);
	}
	return out;
}

/**
 * @brief Apply a given mask to a 3 channel image
 *
 * @param image the 3 channel image in input
 * @param mask the mask to apply (IMPORTANT grayscale, <unsigned char>)
 * @return cv::Mat the 3 channel image with the applied mask
 */
cv::Mat applyMask(cv::Mat &image, cv::Mat &mask){
	cv::Mat out = image.clone();
	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++){
			if (mask.at<unsigned char>(i, j) < 250){
				out.at<cv::Vec3b>(i, j)[0] = 0;
				out.at<cv::Vec3b>(i, j)[1] = 0;
				out.at<cv::Vec3b>(i, j)[2] = 0;
			}
		}
	}
	return out;
}

/**
 * @brief Given a map of images and a map of masks: apply the mask to the corresponding image
 *
 * @param images map <int, cv::Mat> (list of images with corresponding index)
 * @param masks map <int, cv::Mat> (list of masks with corresponding index)
 * @return std::map<int, cv::Mat> the map containing images with the applied mask
 */
std::map<int, cv::Mat> applyMaskMap(std::map<int, cv::Mat> images, std::map<int, cv::Mat> masks){
	std::map<int, cv::Mat> out;
	cv::Mat tmp;
	for (int i = 0; i < images.size(); i++){
		cv::cvtColor(masks[i], tmp, cv::COLOR_BGR2GRAY);
		out[i] = applyMask(images[i], tmp);
	}
	return out;
}


void extractKeypointsWithMask(cv::Mat const& image, cv::Mat const& mask, std::vector<cv::KeyPoint> &outKeyPoints,
							  cv::Mat& outDescriptors, std::vector<cv::KeyPoint>& negKeyPoints, cv::Mat& negDescriptors){
	std::vector<cv::KeyPoint> newKeyPoints;
	cv::Mat newDescriptors;
	std::vector<cv::KeyPoint> newNegKeyPoints;

	extractKeypoints(image, outKeyPoints, outDescriptors);
	if(outKeyPoints.size() == 0){
		//if there are no keypoints detected 
		return;
	}

	std::vector<std::vector<float>> vecDesc, newVecDesc, newVecNegDesc;
	vecDesc = mat2vec<float>(outDescriptors);

	for(int i=0; i<outKeyPoints.size(); i++){
		if(mask.at<cv::Vec3b>(outKeyPoints[i].pt.y, outKeyPoints[i].pt.x)[0]>250){
			newKeyPoints.push_back(outKeyPoints[i]);
			newVecDesc.push_back(vecDesc[i]);
		}else{
			newNegKeyPoints.push_back(outKeyPoints[i]);
			newVecNegDesc.push_back(vecDesc[i]);
		}
	}
	if(newKeyPoints.size() == 0){
		negKeyPoints = newNegKeyPoints;
		negDescriptors = vec2mat<float>(newVecNegDesc, CV_32F);
		return;
	}
	if(newNegKeyPoints.size() == 0){
		outKeyPoints = newKeyPoints;
		outDescriptors = vec2mat<float>(newVecDesc, CV_32F);
		return;
	}
	negKeyPoints = newNegKeyPoints;
	negDescriptors = vec2mat<float>(newVecNegDesc, CV_32F);
	outKeyPoints = newKeyPoints;
	outDescriptors = vec2mat<float>(newVecDesc, CV_32F);
}

/**
 * @brief Given a folder path: load images in a map structure with key the mask name.
 *
 * @param pathDir path of the folder
 * @return std::map<int, cv::Mat> the map with <img, cv::Mat>
 */
std::map<int, cv::Mat> loadImgFromPath(std::string pathDir){
	std::string path = pathDir;
	std::map<int, cv::Mat> images;
	cv::Mat readImg;
	std::string p;
	for (const auto &entry : std::filesystem::directory_iterator(path)){
		p = entry.path().string();
		readImg = (cv::imread(p));
		p = p.substr(path.length() + 1, p.length() - (path.length() + 1) - 4);
		images[std::stoi(p)] = readImg.clone();
	}
	std::cout << "Read " << images.size() << " images from " << pathDir << std::endl;
	return images;
}

/**
 * @brief Load the images inside a folder resizing them to a defined size
 *
 * @param The path of the folder containing the images and the dimensions of the output images
 * @return std::map<int, std::vector<cv::Mat>> a map between the images their ID 
 */
std::map<int, cv::Mat> loadImgFromPath(std::string pathDir, int rows, int cols){
	std::string path = pathDir;
	std::map<int, cv::Mat> images;
	cv::Mat readImg;
	std::string p;
	for (const auto &entry : std::filesystem::directory_iterator(path))	{
		p = entry.path().string();
		readImg = (cv::imread(p));
		cv::resize(readImg, readImg, cv::Size(cols, rows), cv::INTER_LINEAR);
		p = p.substr(path.length() + 1, p.length() - (path.length() + 1) - 4);
		// TEST std::cout << p << std::endl;
		images[std::stoi(p)] = readImg.clone();
	}
	std::cout << "Read " << images.size() << " images from " << pathDir << std::endl;
	return images;
}

/**
 * @brief Read the bounding boxes needed to assess the precision of the system
 *
 * @param The path of the folder containing the annotations
 * @return std::map<int, std::vector<cv::Rect>> a map between the ID of the image and the bounding boxes annotated
 */
std::map<int, std::vector<cv::Rect>> loadBboxesFromPath(std::string pathDir){
	std::string path = pathDir;
	std::string p;	
	std::string line;
	std::map<int, std::vector<cv::Rect>> bboxes;
	cv::Rect act_rect;
	for (const auto &entry : std::filesystem::directory_iterator(path)){
		p = entry.path().string();
		std::ifstream myfile(p);
		p = p.substr(path.length() + 1, p.length() - (path.length() + 1) - 4);
		if (myfile.is_open()){
			while (std::getline(myfile, line)){
				std::stringstream text_of_line(line);
				text_of_line >>act_rect.x >> act_rect.y>>act_rect.width>> act_rect.height;
				bboxes[std::stoi(p)].push_back(act_rect);
			}
			myfile.close();
		}
	}
	return bboxes;
}

/**
 * @brief Generate the window in which keypoint matching has to be processed
 *
 * @param The image, the coordinates to start from, the size of the rectangle to generate
 * @return cv::Mat an image representing the window requested by the arguments
 */
cv::Mat getWindow(cv::Mat img, int row, int col, int rectRows, int rectCols){
	cv::Mat win(rectRows, rectCols, CV_8UC3);
	for (int i = 0; i < rectRows; i++){
		for (int j = 0; j < rectCols; j++){
			win.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(i + row, j + col);
		}
	}
	return win;
}

/**
 * @brief Performs the segmentation colouring the hands inside the rectangles of an image
 *
 * @param The image, the rectangles
 * @return cv::Mat the image coloured where there are hands
 */
cv::Mat getSegmentation(cv::Mat input_image, std::vector<cv::Rect> bounding_boxes) {
	cv::Mat segmentation_mask = cv::Mat(input_image.rows, input_image.cols, CV_8UC1, cv::Scalar(0));
	cv::Mat segmented_image = morphMask(input_image);
	srand(time(NULL));
	int r, g, b;
	for (int i = 0; i < bounding_boxes.size(); i++) {
		r = rand() % 256;
		g = rand() % 256;
		b = rand() % 256;
		for (int x = bounding_boxes[i].x; x < bounding_boxes[i].x + bounding_boxes[i].width; x++) {
			for (int y = bounding_boxes[i].y; y < bounding_boxes[i].y + bounding_boxes[i].height; y++) {
				if (segmented_image.at<cv::Vec3b>(y, x)[0] != 0 || segmented_image.at<cv::Vec3b>(y, x)[1] != 0 || segmented_image.at<cv::Vec3b>(y, x)[2] != 0) {
					segmentation_mask.at<uchar>(y, x) = 255;
					input_image.at<cv::Vec3b>(y, x)[0] = b;
					input_image.at<cv::Vec3b>(y, x)[1] = g;
					input_image.at<cv::Vec3b>(y, x)[2] = r;
				}
			}
		}
	}
	return segmentation_mask;
}


/**
 * @brief Resizes a rectangle that was calculated for a different dimension of the image 
 *
 * @param A rectangle, the size in which the rectangle has been calculated, the original size of the image
 * @return cv::Rect A rectangle resized to be applied into the original image
 */
cv::Rect resizeRect(cv::Rect r, cv::Size previous_size, cv::Size new_size) {
	double width_ratio = (double)(new_size.width) / (double)(previous_size.width);
	double height_ratio = (double)(new_size.height) / (double)(previous_size.height);
	cv::Rect out_Rect;
	//Resize parameters basing on proportions between heights or widths
	out_Rect.x = (int)(r.x * width_ratio);
	out_Rect.y = (int)(r.y * height_ratio);
	out_Rect.width = (int)(r.width * width_ratio);
	out_Rect.height = (int)(r.height * height_ratio);
	return out_Rect;
}
