#ifndef DETECTIONKMEANS_H_INCLUDED

void extractAndSaveDescriptorsKMeans(std::map<int, cv::Mat> images, std::map<int, cv::Mat> masks, cv::Mat& positiveDescriptors, cv::Mat& negativeDescriptors);
std::vector<cv::Rect> slidingWindowDescriptors(const cv::Mat& img, const cv::Mat& positiveDescriptors, const cv::Mat& negativeDescriptors,
	int rectRows, int rectCols, float th, int step);

cv::Mat computeHistogram(const cv::Mat& image, const cv::Mat& descriptors);
void displayHistogram(const cv::Mat histogram);

#endif
