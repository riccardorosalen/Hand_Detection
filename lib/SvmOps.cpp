#include <opencv2/opencv.hpp>
#include "SvmOps.h"
#include "ImgOps.h"
#include "Utility.h"


void dataAndLabels(std::map<int, cv::Mat>& images, std::map<int, cv::Mat>& masks, cv::Mat& data, cv::Mat& lab) {

    //--------------------| Extract keypoints |----------------------------------------------------
    std::vector<std::vector<float>> descriptors;
    std::vector<cv::KeyPoint> key_points;
    std::vector<std::vector<float>> neg_descriptors;
    std::vector<cv::KeyPoint> neg_key_points;
    extractVecDescriptorsMask(images, masks, key_points, descriptors, neg_key_points, neg_descriptors);
    std::cout << "descriptors extracted" << std::endl;


    //--------------------| Decrease the number of keypoints using k-means |-----------------------
    //descriptors = kMeans(1000, descriptors, 10, 0.5);
    //neg_descriptors = kMeans(1000, neg_descriptors, 10, 0.5);


    //--------------------| Create training set |--------------------------------------------------
    //descriptors = normalizeVecVec(descriptors, 255.0);
    //neg_descriptors = normalizeVecVec(neg_descriptors, 255.0);


    cv::Mat descriptors_mat = vec2mat<float>(descriptors, CV_32F);
    cv::Mat neg_descriptors_mat = vec2mat<float>(neg_descriptors, CV_32F);
    cv::Mat positive_labels(descriptors.size(), 1, CV_32S);
    cv::Mat negative_labels(neg_descriptors.size(), 1, CV_32S);
    positive_labels.setTo(1);
    negative_labels.setTo(2);

    cv::Mat trainData;
    cv::Mat labels;

    cv::vconcat(descriptors_mat, neg_descriptors_mat, trainData);
    //trainData = trainData/255.0;
    cv::vconcat(positive_labels, negative_labels, labels);


    data = trainData;
    lab = labels;
    std::cout << "train rows: " << trainData.rows << " train cols: " << trainData.cols << std::endl;
    std::cout << "positive kp: " << descriptors.size() << " negative kp: " << neg_descriptors.size() << std::endl;
}

void normalizeData(cv::Mat& data, cv::Mat& m, cv::Mat& s) {
    //Normalize data
    cv::Mat means, sigmas;  //matrices to save all the means and standard deviations
    for (int i = 0; i < data.cols; i++) {  //take each of the features in vector
        cv::Mat mean; cv::Mat sigma;
        meanStdDev(data.col(i), mean, sigma);  //get mean and std deviation
        means.push_back(mean);
        sigmas.push_back(sigma);
        data.col(i) = (data.col(i) - mean) / sigma;  //normalization
    }
    m = means;
    s = sigmas;
    cv::Mat meansigma;
    cv::hconcat(means, sigmas, meansigma);
    cv::imwrite("normalizationParams.tiff", meansigma);
}

void normalizeDataGivenParams(cv::Mat& data, cv::Mat& means, cv::Mat& sigmas) {
    for (int i = 0; i < data.cols; i++) {
        data.col(i) = (data.col(i) - means.at<float>(i)) / sigmas.at<float>(i);
    }
}
void normalizeDataGivenParams(cv::Mat& data, const std::string path) {
    cv::Mat meansigma = cv::imread(path);
    cv::Mat means = meansigma.col(0);
    cv::Mat sigmas = meansigma.col(1);
    for (int i = 0; i < data.cols; i++) {
        data.col(i) = (data.col(i) - means.at<float>(i)) / sigmas.at<float>(i);
    }
}

void trainAndSaveSVM(const cv::Mat& train_data, const cv::Mat& labels, int iterations, int kernel) {
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

    svm->setType(cv::ml::SVM::C_SVC);
    svm->setDegree(2);
    svm->setC(0.1);
    svm->setKernel(kernel);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, iterations, 1e-6));

    std::vector<std::vector<float>> classweight = { {1}, {1.1} };
    /*     classweight.push_back(std::vector<float>(0));
        classweight.push_back(std::vector<float>(0));
        classweight[0].push_back(1);
        classweight[1].push_back(1.1); */
    cv::Mat matClassWeight = vec2mat<float>(classweight, CV_32F);
    std::cout << "rows: " << matClassWeight.rows << " cols: " << matClassWeight.cols << std::endl;
    svm->setClassWeights(matClassWeight);

    //--------------------| Train the SVM |--------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    svm->train(train_data, cv::ml::ROW_SAMPLE, labels);
    //svm->trainAuto(train_data, cv::ml::ROW_SAMPLE, labels, 10, c_params, gamma_params, zero, zero, zero, zero);

    auto finish = std::chrono::high_resolution_clock::now();
    auto s = std::chrono::duration_cast<std::chrono::seconds>(finish - start);
    std::cout << "Finished training process in " << s.count() << "s" << std::endl;

    //--------------------| Save the SVM |---------------------------------------------------------
    svm->save("../SVM_model.txt");

    //--------------------| Show support vectors |-------------------------------------------------
    cv::Mat I = cv::Mat::zeros(512, 512, CV_8UC3);
    int thick = 2;
    cv::Mat sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; i++)
    {
        const float* v = sv.ptr<float>(i);
        cv::circle(I, cv::Point((int)v[0], (int)v[1]), 6, cv::Scalar(128, 128, 128), thick);
    }

    cv::imshow("SVM for Non-Linear Training Data", I); // show it to the user
    cv::waitKey();
}

void svmEvaluateOneImage(const cv::Mat& image, const cv::Mat& mask, const std::string path) {
    cv::Ptr<cv::ml::SVM> svm = cv::Algorithm::load<cv::ml::SVM>(path);

    //--------------------| Extract key points and descriptors from an image |---------------------
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    int id_image = 2;
    extractKeypoints(image, kp, desc);
    std::vector<std::vector<float>> vecDesc = mat2vec<float>(desc);

    std::cout << "Keypoints extracted: " << kp.size() << std::endl;

    //--------------------| Predict with SVM the class of the point |------------------------------
    for (int i = vecDesc.size() - 1; i >= 0; i--) {
        if (svm->predict(vecDesc[i]) == 2) {
            kp.erase(kp.begin() + i);
            vecDesc.erase(vecDesc.begin() + i);
        }
    }
    std::cout << "Class predicted, class 1 size: " << kp.size() << std::endl;

    //
    int counter = 0;
    for (int i = 0; i < kp.size(); i++) {
        if ((mask.at<cv::Vec3b>(kp[i].pt.y, kp[i].pt.x)[0] > 100)) {
            counter = counter + 1;
        }
    }
    std::cout << "Giusti: " << counter << '/' << kp.size() << " ~~~~~~ " << (float)counter / kp.size() << std::endl;

    //--------------------| Display output image with key points |---------------------------------
    cv::Mat out;
    cv::drawKeypoints(image, kp, out);

    cv::namedWindow("SPEREMO");
    cv::imshow("SPEREMO", out);
    cv::waitKey(0);
}

std::vector<float> evaluateSVM(const cv::Mat& data, const cv::Mat& labels, const std::string path) {
    cv::Ptr<cv::ml::SVM> svm = cv::Algorithm::load<cv::ml::SVM>(path);
    float true_positive = 0;
    float true_negative = 0;
    float false_positive = 0;
    float false_negative = 0;
    int pred;
    for (int i = 0; i < data.rows; i++) {
        pred = svm->predict(data.row(i));
        if (labels.at<int>(i, 0) == 1 && pred == labels.at<int>(i, 0)) {
            true_positive += 1;
        }
        else if (labels.at<int>(i, 0) == 1 && pred != labels.at<int>(i, 0)) {
            false_negative += 1;
        }
        else if (labels.at<int>(i, 0) == 2 && pred == labels.at<int>(i, 0)) {
            true_negative += 1;
        }
        else {
            false_positive += 1;
        }
    }
    float precision = true_positive / (true_positive + false_positive);
    float recall = true_positive / (true_positive + false_negative);
    float accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative);
    float f1 = (precision * recall) / (precision + recall);
    std::cout << "True positive: " << true_positive << ", True negative: " << true_negative
        << ", False positive: " << false_positive << ", False negative: " << false_negative << std::endl;
    std::cout << "Precision tp/tp+fp: " << precision << std::endl;
    std::cout << "Recall tp/tp+fn: " << recall << std::endl;
    std::cout << "Accuracy: " << accuracy << std::endl;
    std::cout << "F1: " << f1 << std::endl;
    std::vector<float> out;
    out.push_back(true_positive);
    out.push_back(true_negative);
    out.push_back(false_positive);
    out.push_back(false_negative);
    out.push_back(precision);
    out.push_back(recall);
    out.push_back(accuracy);
    out.push_back(f1);
    return out;
}


std::vector<cv::Rect> slidingWindowSVM(const cv::Mat& img, int rect_rows, int rect_cols, const std::string path, float th) {

    cv::Ptr<cv::ml::SVM> svm = cv::Algorithm::load<cv::ml::SVM>(path);

    std::vector<cv::Rect> out;
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    int good_kps_count;
    std::vector<cv::KeyPoint> good_kps;

    //std::cout << img.rows << " " << img.cols << std::endl;
    for (int i = 0; i < img.rows - rect_rows; i = i + 20) {

        for (int j = 0; j < img.cols - rect_cols; j = j + 20) {
            cv::Mat win = getWindow(img, i, j, rect_cols, rect_rows);

            extractKeypoints(win, kps, desc);


            good_kps_count = 0;
            good_kps.clear();
            for (int i = 0; i < desc.rows; i++) {
                if (svm->predict(desc.row(i)) == 1) {
                    good_kps_count++;
                    good_kps.push_back(kps[i]);
                }
            }
            /*    cv::drawKeypoints(win, good_kps, win);
               cv::imshow("", win);
               cv::waitKey(); */
            if (kps.size() > 0) {
                if (((float)good_kps_count / kps.size()) > th) {
                    //if(good_kps_count>th){
                    std::cout << (float)good_kps_count / kps.size() << ", ";
                    out.push_back(cv::Rect(j, i, rect_cols, rect_rows));
                }
            }
        }
    }
    return out;
}