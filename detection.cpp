#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
//#include <experimental/random>


#include "lib/ImgOps.h"
#include "lib/Utility.h"
#include "lib/SvmOps.h"
#include "lib/HOG.h"
#include "lib/PreProcessing.h"
#include "lib/RectClustering.h"
#include "lib/Metrics.h"
#include "lib/detectionKmeans.h"



int main(int argc, char** argv) {

    std::string path_to_data_dir=argv[1];
    //--------------------| Read Images |----------------------------------------------------------
    std::map<int, cv::Mat> finalImages = loadImgFromPath(path_to_data_dir+"/rgb", 216, 384);
    std::map<int, cv::Mat> finalMasks = loadImgFromPath(path_to_data_dir+"/mask", 216, 384);
    std::map<int, std::vector<cv::Rect>> finalBboxes = loadBboxesFromPath(path_to_data_dir+"/det");

    std::map<int, cv::Mat> images = loadImgFromPath(path_to_data_dir+"/rgb");
    std::map<int, cv::Mat> masks = loadImgFromPath(path_to_data_dir+"/mask");
    std::map<int, std::vector<cv::Rect>> bboxes = loadBboxesFromPath(path_to_data_dir+"/det");

    int dilation_size = 2;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    for (int i = 1; i < masks.size(); i++) {
        cv::dilate(masks[i], masks[i], element);
    }

    cv::Mat positive_descriptors;
    cv::Mat negative_descriptors;
    cv::FileStorage file("../data/positiveDescriptorsKMeans.csv", cv::FileStorage::READ);
    file["positive"] >> positive_descriptors;
    file.release();
    cv::FileStorage file2("../data/negativeDescriptorsKMeans.csv", cv::FileStorage::READ);
    file2["negative"] >> negative_descriptors;
    file2.release();


    //--------------------| Train the SVM |--------------------------------------------------------
    /* cv::Mat trainData, labels;
    dataAndLabels(images, masks, trainData, labels);
    cv::Mat means, sigmas;
    normalizeData(trainData, means, sigmas);
    trainAndSaveSVM(trainData, labels, 300000, cv::ml::SVM::POLY); */



    //--------------------| Evaluate SVM model |---------------------------------------------------
    //std::string modelPath = "../svmModels/SVM_model_12.txt";
/*     std::string modelPath = "../SVM_model.txt";
    cv::Mat testData, testLabels;
    dataAndLabels(testImages, testMasks, testData, testLabels);
    normalizeDataGivenParams(testData, means, sigmas);
    std::cout << "Model evaluation on train data" << std::endl;
    evaluateSVM(trainData, labels, modelPath);
    std::cout << "Model evaluation on test data" << std::endl;
    evaluateSVM(testData, testLabels, modelPath); */

    HOGwSVM hgsvm;
    hgsvm.load_trained_detector("../data/detector.txt");
    //Serve?
    //matSize(images[i]);

    int images_to_compute=20;
    cv::Mat pre_process_img;
    std::vector<cv::Rect> kmeans_detector_rects;
    std::vector<cv::Rect> hog_rects;
    std::vector<cv::Rect> small_rects;
    double clustering_tresh = 0.01;
    double rect_volume_tresh = 0.15;
    std::vector<cv::Rect> cluster_centroids;
    std::vector<std::vector<cv::Rect>> images_bounding_boxes;
    std::vector<float> ImagesIOU;
    float average_IOU=0;
    for (int i = 1; i <=images_to_compute; i++) {
        std::cout<<"Starting image:"<<i<<std::endl;
        pre_process_img = morphMask(skinSeg(images[i]));
        kmeans_detector_rects =slidingWindowDescriptors(finalImages[i], positive_descriptors, negative_descriptors, 50, 50, 30, 15);
        hog_rects = hgsvm.detectRect(pre_process_img);

        for (int j = 0; j < kmeans_detector_rects.size(); j++) {
            kmeans_detector_rects[j] = resizeRect(kmeans_detector_rects[j], finalImages[i].size(), images[i].size());
            if (isGoodRect(pre_process_img, kmeans_detector_rects[j])) {
                std::vector<cv::Rect> splitted = split2(kmeans_detector_rects[j]);
                for (int l = 0; l < splitted.size(); l++) {
                    if (isGoodRect(pre_process_img, splitted[l])) {
                        small_rects.push_back(splitted[l]);
                    }
                }
            }
        }
        for (int j = 0; j < hog_rects.size(); j++) {
            cv::Scalar color = cv::Scalar(255, 0, 0);
            if (isGoodRect(pre_process_img, hog_rects[j])) {
                std::vector<cv::Rect> splitted = split2(hog_rects[j]);
                for (int l = 0; l < splitted.size(); l++) {
                    if (isGoodRect(pre_process_img, splitted[l])) {
                        small_rects.push_back(splitted[l]);
                    }
                }

            }
        }
        std::vector<std::vector<cv::Rect>>  clusters = cluster_rects(small_rects, clustering_tresh);
        for (int k = 0; k < clusters.size(); k++) {
            if (100 * union_of_rects(clusters[k]).area() / (images[i].rows * images[i].cols) >rect_volume_tresh) {
                cluster_centroids.push_back(union_of_rects(clusters[k]));
                cv::rectangle(images[i], cluster_centroids[k], cv::Scalar(0, 255, 0));
            }
            
        }
        images_bounding_boxes.push_back(cluster_centroids);
        
        kmeans_detector_rects.clear();
        hog_rects.clear();
        small_rects.clear();
        cluster_centroids.clear();
        std::cout << "Image " << i<<" IOU:" << iouMetric(bboxes[i], images_bounding_boxes[i-1] )<< std::endl;
        average_IOU = average_IOU + iouMetric( bboxes[i], images_bounding_boxes[i-1]);
        std::cout<<"End of image:"<<i<<std::endl;
    }
    for(int i=1;i<=images.size();i++){
        cv::imshow("Image", images[i]);
        cv::waitKey(0);
    }
    average_IOU=average_IOU/images_to_compute;
    std::cout << "Average IOU of the System:" << average_IOU << std::endl;
    return 0;

}
