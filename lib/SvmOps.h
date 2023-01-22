#ifndef SVMOPS_H_INCLUDED

void dataAndLabels(std::map<int, cv::Mat>& images, std::map<int, cv::Mat>& masks, cv::Mat& data, cv::Mat& lab);

void normalizeData(cv::Mat& data, cv::Mat& m, cv::Mat& s);
void normalizeDataGivenParams(cv::Mat& data, cv::Mat& means, cv::Mat& sigmas);
void normalizeDataGivenParams(cv::Mat& data, const std::string path);

void trainAndSaveSVM(const cv::Mat& train_data, const cv::Mat& labels, int iterations, int kernel);

void svmEvaluateOneImage(const cv::Mat& image, const cv::Mat& mask, const std::string path);

std::vector<float> evaluateSVM(const cv::Mat& data, const cv::Mat& labels, const std::string path);

std::vector<cv::Rect> slidingWindowSVM(const cv::Mat& img, int rect_rows, int rect_cols, const std::string path, float th);


#endif