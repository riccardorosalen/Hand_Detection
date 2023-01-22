#ifndef IMGOPS_H_INCLUDED

#include<iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <map>

std::map<int, cv::Mat> loadImgFromPath(std::string pathDir);
std::map<int, cv::Mat> loadImgFromPath(std::string pathDir, int rows, int cols);
std::map<int, std::vector<cv::Rect>> loadBboxesFromPath(std::string pathDir);
cv::Mat loadFromFile(std::string name);

cv::Mat extractKeypoints(cv::Mat const& image, std::vector<cv::KeyPoint> &outKeyPoints);
cv::Mat extractKeypoints(cv::Mat const& image, std::vector<cv::KeyPoint>& outKeyPoints, cv::Mat& outDescriptors);

void extractKeypointsWithMask(cv::Mat const& image, cv::Mat const& mask, std::vector<cv::KeyPoint> &outKeyPoints,
    cv::Mat& outDescriptors, std::vector<cv::KeyPoint>& negKeyPoints, cv::Mat& negDescriptors);

cv::Mat applyMask(cv::Mat& image, cv::Mat& mask);
std::map<int, cv::Mat> applyMaskMap(std::map<int, cv::Mat> images, std::map<int, cv::Mat> masks);







cv::Mat getWindow(cv::Mat img, int row, int col, int rectRows, int rectCols);

cv::Rect resizeRect(cv::Rect r, cv::Size original_size, cv::Size new_size);
cv::Mat getSegmentation(cv::Mat input_image, std::vector<cv::Rect> bounding_boxes);


#endif
