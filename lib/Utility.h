#ifndef UTILITY_H_INCLUDED

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

double computeL2Distance(std::vector<float> desc1, std::vector<float> desc2);
double computeL1Distance(std::vector<float> desc1, std::vector<float> desc2);

std::vector<std::vector<float>> kMeans(int k, std::vector<std::vector<float>> points, int max_iter, float threshold);

std::vector<std::vector<float>> extractVecDescriptors(std::map<int, cv::Mat>& hand_segmented);
void extractVecDescriptorsMask(std::map<int, cv::Mat>& images, std::map<int, cv::Mat>& masks, std::vector<cv::KeyPoint>& kps, std::vector<std::vector<float>>& descs);
void extractVecDescriptorsMask(std::map<int, cv::Mat>& images, std::map<int, cv::Mat>& masks, std::vector<cv::KeyPoint>& out_kp,
    std::vector<std::vector<float>>& out_desc, std::vector<cv::KeyPoint>& out_neg_kp, std::vector<std::vector<float>>& out_neg_desc);
std::vector<std::vector<float>> extractVecDescriptorsMask(std::map<int, cv::Mat>& hand_segmented, std::map<int, cv::Mat>& masks);

bool isGoodRect(const cv::Mat& img, cv::Rect rect);
void matSize(const cv::Mat& mat);



template<typename T>
cv::Mat vec2mat(std::vector<std::vector<T>> const& vec, int cvMatType){
    cv::Mat out = cv::Mat(vec.size(), vec[0].size(), cvMatType);
    for(int i=0; i<vec.size(); i++){
        for(int j=0; j<vec[0].size(); j++){
            out.at<T>(i, j) = vec[i][j];
        }
    }
    return out;
}

template<class T>
std::vector<std::vector<T>> mat2vec(cv::Mat const& mat){
    std::vector<std::vector<T>> out;
    for(int i=0; i<mat.rows; i++){
        out.push_back(std::vector<T>());
        for(int j=0; j<mat.cols; j++){
            out[i].push_back(mat.at<T>(i,j));
        }
    }
    return out;
}
#endif
