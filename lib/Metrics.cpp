#include "Metrics.h"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
/**
 * @brief Returns the IOU error given two sets of rectangles
 *
 * @param true_rect the true set of rectangles
 * @param pred_rect the predicted set of rectangles
 * @return float the IOU error
 */
float iouMetric(std::vector<cv::Rect> true_rect, std::vector<cv::Rect> pred_rect) {
    float error = 0;
    double temp_value, intersection_value, union_value=0;
    cv::Rect rect_intersection;
    for (int j = 0; j < pred_rect.size(); j++) {
            union_value=union_value+pred_rect[j].area();
    }
    for (int i = 0; i < true_rect.size(); i++) {
        union_value=union_value+true_rect[i].area();
        temp_value = 0;
        for (int j = 0; j < pred_rect.size(); j++) {
            rect_intersection = true_rect[i] & pred_rect[j];
            intersection_value = intersection_value+rect_intersection.area();
        }
    }
    return (float)(intersection_value/union_value) ;
}

/**
 * @brief Calculates pixel axxuracy using IOU method given 2 masks in 3 channel (for simplicity of this project)
 * 
 * @param true_mask the mask of true pixels
 * @param pred_mask the mask of predicted pixels
 * @return std::vector<float> vector of size 2 containing first the iou of the skin region and as second element the iou of non skin region.
 */
std::vector<float> pixelAccuracyMetric(cv::Mat& true_mask, cv::Mat& pred_mask) {
    if ((true_mask.rows != pred_mask.rows) || (true_mask.cols != pred_mask.cols)) {
        std::cout << "ERROR(segmentationMetric): true_mask and pred_mask have not the same size!" << std::endl;
    }
    cv::cvtColor(true_mask,true_mask,cv::COLOR_RGB2GRAY,0);
    float area_of_overlap_hand = 0;
    float area_of_overlap_non_hand = 0;
    float area_of_union_hand = 0;
    float area_of_union_non_hand = 0;

    for (int i = 0; i < true_mask.rows; i++) {
        for (int j = 0; j < true_mask.cols; j++) {
            if (true_mask.at<uchar>(i, j)==255) {
                area_of_union_hand += 1;
                if (pred_mask.at<uchar>(i, j)==255) {
                    area_of_overlap_hand += 1;
                }
            }
            else {
                area_of_union_non_hand += 1;
                if (pred_mask.at<uchar>(i, j)==0) {
                    area_of_overlap_non_hand += 1;
                }
            }

            if (pred_mask.at<uchar>(i, j) ==255) {
                area_of_union_hand += 1;
            }
            else {
                area_of_union_non_hand += 1;
            }
        }
    }
    return { (area_of_overlap_hand / (area_of_union_hand - area_of_overlap_hand)), (area_of_overlap_non_hand / (area_of_union_non_hand - area_of_overlap_non_hand)) };
}
