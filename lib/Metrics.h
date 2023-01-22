#ifndef METRICS_H_INCLUDED
#include<opencv2/highgui.hpp>

float iouMetric(std::vector<cv::Rect> trueRect, std::vector<cv::Rect> predRect);
std::vector<float> pixelAccuracyMetric(cv::Mat& true_mask, cv::Mat& pred_mask);

#endif
