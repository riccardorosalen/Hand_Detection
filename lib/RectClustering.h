#ifndef RECTCLUSTERING_H_INCLUDED

#include<opencv2/highgui.hpp>
#include<iostream>

//Merges all the rectangles in a cluster in a single one
cv::Rect union_of_rects(const std::vector<cv::Rect>& cluster);
//Split into smaller squares/rectangles a big one
std::vector<cv::Rect> split2(cv::Rect r);
//Clusters rectangles toghether if they interstect more than a t
std::vector<std::vector<cv::Rect>> cluster_rects(const std::vector<cv::Rect>& rects, const double threshold);
#endif

