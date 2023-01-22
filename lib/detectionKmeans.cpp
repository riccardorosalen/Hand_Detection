#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
//#include <experimental/random>

#include "PreProcessing.h"
#include "ImgOps.h"
#include "Utility.h"
#include "SvmOps.h"
#include "HOG.h"
#include "detectionKmeans.h"


void extractAndSaveDescriptorsKMeans(std::map<int, cv::Mat> images, std::map<int, cv::Mat> masks, cv::Mat& positiveDescriptors, cv::Mat& negativeDescriptors){
    cv::Mat trainData, trainLabels;
    dataAndLabels(images, masks, trainData, trainLabels);
    cv::Mat clusterLabels, clusterCenters;
    matSize(trainData);
    cv::TermCriteria tc = cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10, 0.5);
    cv::kmeans(trainData, 1000, clusterLabels, tc, 3, cv::KmeansFlags::KMEANS_PP_CENTERS, clusterCenters);

    matSize(clusterLabels);
    matSize(clusterCenters);

    for(int i=0; i<20; i++){
        std::cout << clusterLabels.at<int>(i) << std::endl;
    }
    cv::Mat clusterCount = cv::Mat(clusterCenters.rows, 2, CV_32F);
    clusterCount.setTo(0);

    for(int i=0; i<trainLabels.rows; i++){
        if(trainLabels.at<int>(i) == 1){
            clusterCount.at<float>(clusterLabels.at<int>(i),0) += 1;
        }else{
            clusterCount.at<float>(clusterLabels.at<int>(i),1) += 1;
        }
    }
    int count = 0;
    for(int i=0; i<clusterCount.rows; i++){
        if(clusterCount.at<float>(i,0)/clusterCount.at<float>(i,1) > 0.38){
            std::cout << clusterCount.at<float>(i,0) << ", " << clusterCount.at<float>(i,1) << std::endl;
            std::cout << clusterCount.at<float>(i,0)/clusterCount.at<float>(i,1) << std::endl;
            count += 1;
        }
    }
    std::cout << count << std::endl;
    count = 0;
    for(int i=0; i<clusterCount.rows; i++){
        if(clusterCount.at<float>(i,0)/clusterCount.at<float>(i,1) < 0.17){
            std::cout << clusterCount.at<float>(i,0) << ", " << clusterCount.at<float>(i,1) << std::endl;
            std::cout << clusterCount.at<float>(i,0)/clusterCount.at<float>(i,1) << std::endl;
            count += 1;
        }
    }
    std::cout << count << std::endl;


    cv::Mat positiveDescriptorsKMeans;
    for(int i=0; i<clusterCount.rows; i++){
        if(clusterCount.at<float>(i,0)/clusterCount.at<float>(i,1) > 0.38){
            positiveDescriptorsKMeans.push_back(clusterCenters.row(i));
        }
    }
    positiveDescriptors = positiveDescriptorsKMeans;
    cv::FileStorage file("../positiveDescriptorsKMeans.csv", cv::FileStorage::WRITE);
    file << "positive" << positiveDescriptorsKMeans;
    file.release();

    cv::Mat negativeDescriptorsKMeans;
    for(int i=0; i<clusterCount.rows; i++){
        if(clusterCount.at<float>(i,0)/clusterCount.at<float>(i,1) < 0.17){
            negativeDescriptorsKMeans.push_back(clusterCenters.row(i));
        }
    }
    negativeDescriptors = negativeDescriptorsKMeans;
    file = cv::FileStorage("../negativeDescriptorsKMeans.csv", cv::FileStorage::WRITE);
    file << "negative" << negativeDescriptorsKMeans;
    file.release();
    
}

std::vector<cv::Rect> slidingWindowDescriptors(const cv::Mat& img, const cv::Mat& positiveDescriptors, const cv::Mat& negativeDescriptors, 
int rectRows, int rectCols, float th, int step) {

    std::vector<cv::Rect> out;

    cv::Mat totalDescriptors;
    cv::vconcat(positiveDescriptors, negativeDescriptors, totalDescriptors);
	for (int i = 0; i < img.rows - rectRows; i = i + step) {

		for (int j = 0; j < img.cols - rectCols; j = j + step) {
			cv::Mat win = getWindow(img, i, j, rectCols, rectRows);
            cv::Rect tmp(j, i, rectCols, rectRows);
            
            std::vector<cv::KeyPoint> kps;
            cv::Mat desc;
            //kpsFromRect(fullKps, fullDesc, kps, desc, tmp);
            extractKeypoints(win, kps, desc);
            if(kps.size()>0){
                //cv::imshow("Window", win);
                cv::Mat posHist = computeHistogram(win, positiveDescriptors);
                cv::Mat negHist = computeHistogram(win, negativeDescriptors);
                //displayHistogram(posHist);
                //displayHistogram(negHist);
                float error1 = cv::sum(posHist)[0]/positiveDescriptors.rows;
                float error2 = cv::sum(negHist)[0]/negativeDescriptors.rows;
                //std::cout << error1 << ", " << error2 << std::endl;
                

                if ((error2-error1)>th){
                    cv::Rect tmp(j, i, rectCols, rectRows);
                    out.push_back(tmp);
                }
            }
		}
	}
	return out;
}

cv::Mat computeHistogram(const cv::Mat& image, const cv::Mat& descriptors){
    cv::Mat out = cv::Mat(descriptors.rows, 1, CV_32F);
    out.setTo(0);

    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    extractKeypoints(image, kp, desc);
    for(int i=0; i<descriptors.rows; i++){
        for(int j=0; j<desc.rows; j++){
            out.at<float>(i) += computeL2Distance(descriptors.row(i), desc.row(j));
        }
    }
    return out;
}

void displayHistogram(const cv::Mat histogram){
    if(histogram.rows < 1){
        std::cout << "ERROR(displayHistogram): empty histogram!" << std::endl;
    }
    double min, max;
    cv::Point idmin, idmax;
    cv::minMaxLoc(histogram, &min, &max, &idmin, &idmax);
    cv::Mat newHist = histogram/max;
    newHist = newHist * 300;
    cv::Mat display = cv::Mat(300, histogram.rows, CV_8U);
    display.setTo(0);
    for(int i=0; i<display.rows; i++){
        for(int j=0; j<display.cols; j++){
            if((300-i)<newHist.at<float>(j)){
                display.at<unsigned char>(i,j) = 255;
            }
        }
    }
    cv::imshow("Histogram", display);
    cv::waitKey(0);
}

