#include "HOG.h"

#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>  
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>




//void test_trained_detector(String obj_det_filename, String test_dir, String videofilename);
std::vector< float > HOGwSVM::get_svm_detector(const cv::Ptr< cv::ml::SVM >& svm)
{
    // get the support std::vectors
    cv::Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    cv::Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);
    CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
    CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
        (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
    CV_Assert(sv.type() == CV_32F);
    std::vector< float > hog_detector(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
    hog_detector[sv.cols] = (float)-rho;
    return hog_detector;
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void HOGwSVM::convert_to_ml(const std::vector< cv::Mat >& train_samples, cv::Mat& trainData)
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
    cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = cv::Mat(rows, cols, CV_32FC1);
    for (size_t i = 0; i < train_samples.size(); ++i)
    {
        CV_Assert(train_samples[i].cols == 1 || train_samples[i].rows == 1);
        if (train_samples[i].cols == 1)
        {
            transpose(train_samples[i], tmp);
            tmp.copyTo(trainData.row((int)i));
        }
        else if (train_samples[i].rows == 1)
        {
            train_samples[i].copyTo(trainData.row((int)i));
        }
    }
}

//Simple image loaders
void HOGwSVM::load_images(const std::string& dirname, std::vector< cv::Mat >& img_lst, bool showImages = false){
    std::vector< std::string > files;
    cv::glob(dirname, files);
    for (size_t i = 0; i < files.size(); ++i)
    {
        cv::Mat img = cv::imread(files[i]); // load the image
        if (img.empty())
        {
            std::cout << files[i] << " is invalid!" << std::endl; // invalid image, skip it.
            continue;
        }
        if (showImages)
        {
            imshow("image", img);
            cv::waitKey(1);
        }
        img_lst.push_back(img);
    }
}


void HOGwSVM::sample_neg(const std::vector< cv::Mat >& full_neg_lst, std::vector< cv::Mat >& neg_lst, const cv::Size& size){
    cv::Rect box;
    box.width = size.width;
    box.height = size.height;
    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < full_neg_lst.size(); i++)
        if (full_neg_lst[i].cols > box.width && full_neg_lst[i].rows > box.height)
        {
            box.x = rand() % (full_neg_lst[i].cols - box.width);
            box.y = rand() % (full_neg_lst[i].rows - box.height);
            cv::Mat roi = full_neg_lst[i](box);
            neg_lst.push_back(roi.clone());
        }
}
void HOGwSVM::computeHOGs(const cv::Size wsize, const std::vector< cv::Mat >& img_lst, std::vector< cv::Mat >& gradient_lst, bool use_flip){
    cv::HOGDescriptor hog;
    hog.winSize = wsize;
    cv::Mat gray;
    std::vector< float > descriptors;
    for (size_t i = 0; i < img_lst.size(); i++)
    {
        if (img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height)
        {
            cv::Rect r = cv::Rect((img_lst[i].cols - wsize.width) / 2,
                (img_lst[i].rows - wsize.height) / 2,
                wsize.width,
                wsize.height);
            cvtColor(img_lst[i](r), gray, cv::COLOR_BGR2GRAY);
            hog.compute(gray, descriptors, cv::Size(80, 40), cv::Size(0, 0));
            gradient_lst.push_back(cv::Mat(descriptors).clone());
            if (use_flip)
            {
                flip(gray, gray, 1);
                hog.compute(gray, descriptors, cv::Size(80, 40), cv::Size(0, 0));
                gradient_lst.push_back(cv::Mat(descriptors).clone());
            }
        }
    }
}

//Given the name of a file, set the HOGDescriptor (and the SVM detector inside) as the pre-trained version
void HOGwSVM::load_trained_detector(std::string detector_filename) {
    this->hog.load(detector_filename);
}

//Performs detection of rectangles containing hands
std::vector<cv::Rect> HOGwSVM::detectRect(cv::Mat img) {
    std::vector< cv::Rect > detections;
    std::vector< double > foundWeights;
    this->hog.detectMultiScale(img, detections, foundWeights);
    return detections;
}