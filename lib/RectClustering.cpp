#include "RectClustering.h"
#include <opencv2/highgui.hpp>
#include <iostream>

/**
 * @brief Merges a vector of rectangles in the same cluster in an unique one
 *
 * @param A vector of rectangles
 * @return std::vector<cv::Rect> A rectangle created merging the ones passed as parameter
 */
cv::Rect union_of_rects(const std::vector<cv::Rect>& cluster){
	cv::Rect one;
	if (!cluster.empty()){
		one = cluster[0];
		for (const auto& r : cluster) { one |= r; }
	}
	return one;
}


/**
 * @brief Split a Rectangle into 6x6 sub squares or smaller if size not multiple of 6
 *
 * @param A rectangle
 * @return std::vector<cv::Rect> A vector of mini squares or rectangles from the one passed as parameter
 */
std::vector<cv::Rect> split2(cv::Rect r) {
	std::vector<cv::Rect> out;
	int dim = 5;
	int wincr;
	int hincr;
	for (int h = 0; h < r.height; h = h + hincr) {
		for (int w = 0; w < r.width; w = w + wincr) {
			if (r.height - h > dim) {
				hincr = dim;
			}
			else {
				hincr = r.height - h;
			}
			if (r.width - w > dim) {
				wincr = dim;
			}
			else {
				wincr = r.width - w;
			}
			//Bigger than the window size to make them intersect
			cv::Rect act = cv::Rect(r.x + w, r.y + h, wincr + 1, hincr + 1);
			out.push_back(act);
		}
	}
	return out;
}

/**
 * @brief Cluster rectangles that intersect over a certain threshold
 *
 * @param A vector of Rects and the treshold
 * @return std::vector<std::vector<cv::Rect>> The clusters of rects with all the rects in each cluster
 */
std::vector<std::vector<cv::Rect>> cluster_rects(const std::vector<cv::Rect>& rects, const double threshold){
	std::vector<int> labels;
	//Partitions requires as arguments: vector of rects, vector of labels, predicate to evaluate and cluster
	int n_labels = cv::partition(rects, labels, [threshold](const cv::Rect& lhs, const cv::Rect& rhs) {
		double i = static_cast<double>((lhs & rhs).area());
		double ratio_intersection_over_lhs_area = i / static_cast<double>(lhs.area());
		double ratio_intersection_over_rhs_area = i / static_cast<double>(rhs.area());
		return (ratio_intersection_over_lhs_area > threshold) || (ratio_intersection_over_rhs_area > threshold);
		});
	std::vector<std::vector<cv::Rect>> clusters(n_labels);
	for (size_t i = 0; i < rects.size(); ++i) {
		clusters[labels[i]].push_back(rects[i]);
	}
	return clusters;
}