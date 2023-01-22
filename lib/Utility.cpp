//#include <experimental/random> //std::experimental::randint(min, max);
#include <map>
#include "Utility.h"
#include "ImgOps.h"

double computeL2Distance(std::vector<float> desc1, std::vector<float> desc2){
    double sum = 0;
    for(int i=0; i<desc1.size(); i++){
        sum = sum + ((desc1[i]-desc2[i])*(desc1[i]-desc2[i]));
    }
    return std::sqrt(sum);
}
double computeL2DistanceMat(const cv::Mat& desc1, const cv::Mat& desc2){
    double sum = 0;
    float d1, d2;
    for(int i=0; i<desc1.cols; i++){
        d1 = desc1.at<float>(0,i);
        d2 = desc2.at<float>(0,i);
        sum = sum + ((d1-d2)*(d1-d2));
    }
    return std::sqrt(sum);
}

double computeL1Distance(std::vector<float> desc1, std::vector<float> desc2){
    double sum = 0;
    for(int i=0; i<desc1.size(); i++){
        sum = sum + (std::abs(desc1[i])-std::abs(desc2[i]));
    }
    return sum;
}

std::vector<std::vector<float>> extractVecDescriptors(std::map<int, cv::Mat>& hand_segmented){
    int counter = 0;
    std::vector<std::vector<float>> out_desc;
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    for(int k=0; k<hand_segmented.size(); k++){
        extractKeypoints(hand_segmented[k], kp, desc);
        for(int i=0; i<desc.rows; i++){
            out_desc.push_back(std::vector<float>());
            for(int j=0; j<desc.cols; j++){
                out_desc[counter].push_back(desc.at<float>(i,j));
            }
            counter = counter + 1;
        }
    }
    return out_desc;

}

std::vector<std::vector<float>> extractVecDescriptorsMask(std::map<int, cv::Mat>& images, std::map<int, cv::Mat>& masks){
    int counter = 0;
    std::vector<std::vector<float>> out_desc;
    std::vector<cv::KeyPoint> kp, nkp;
    cv::Mat desc, ndesc;
    for(int k=0; k<images.size(); k++){
        extractKeypointsWithMask(images[k], masks[k], kp, desc, nkp, ndesc);
        for(int i=0; i<desc.rows; i++){
            out_desc.push_back(std::vector<float>());
            for(int j=0; j<desc.cols; j++){
                out_desc[counter].push_back(desc.at<float>(i,j));
            }
            counter = counter + 1;
        }
    }
    return out_desc;

}

void extractVecDescriptorsMask(std::map<int, cv::Mat>& images, std::map<int, cv::Mat>& masks, std::vector<cv::KeyPoint>& kps, std::vector<std::vector<float>>& descs){
    int counter = 0;
    std::vector<std::vector<float>> out_desc;
    std::vector<cv::KeyPoint> kp, nkp;
    cv::Mat desc, ndesc;
    for(int k=0; k<images.size(); k++){
        extractKeypointsWithMask(images[k], masks[k], kp, desc, nkp, ndesc);
        for(int i=0; i<desc.rows; i++){
            out_desc.push_back(std::vector<float>());
            for(int j=0; j<desc.cols; j++){
                out_desc[counter].push_back(desc.at<float>(i,j));
            }
            counter = counter + 1;
        }
    }
    
    kps = kp;
    descs = out_desc;

}

void extractVecDescriptorsMask(std::map<int, cv::Mat>& images, std::map<int, cv::Mat>& masks, std::vector<cv::KeyPoint>& out_kp,
std::vector<std::vector<float>>& out_desc, std::vector<cv::KeyPoint>& out_neg_kp, std::vector<std::vector<float>>& out_neg_desc){
    int counter1 = 0;
    int counter2 = 0;
    std::vector<std::vector<float>> descriptors;
    std::vector<std::vector<float>> negDescriptors;
    std::vector<cv::KeyPoint> key_points;
    std::vector<cv::KeyPoint> neg_key_points;
    
    std::vector<cv::KeyPoint> kp;
    std::vector<cv::KeyPoint> neg_kp;
    cv::Mat desc;
    cv::Mat neg_desc;
    for(int k=0; k<images.size(); k++){
        extractKeypointsWithMask(images[k], masks[k], kp, desc, neg_kp, neg_desc);
        for(int i=0; i<desc.rows; i++){
            key_points.push_back(kp[i]);
            descriptors.push_back(std::vector<float>());
            for(int j=0; j<desc.cols; j++){
                descriptors[counter1].push_back(desc.at<float>(i,j));
            }
            counter1 = counter1 + 1;
        }
        for(int i=0; i<neg_desc.rows; i++){
            if(neg_desc.cols == 128){
                neg_key_points.push_back(kp[i]);
                negDescriptors.push_back(std::vector<float>());
                for(int j=0; j<neg_desc.cols; j++){
                    negDescriptors[counter2].push_back(neg_desc.at<float>(i,j));
                }
                counter2 = counter2 + 1;
            }else{
                std::cout << "ERROR, no negative descriptors for image: " << k  << " rows: " << neg_desc.rows << std::endl;
            }
        }
    }
    out_kp = key_points;
    out_desc = descriptors;
    out_neg_kp = neg_key_points;
    out_neg_desc = negDescriptors;

}

bool isGoodRect(const cv::Mat& img, cv::Rect rect){
    //cv::Mat win = img(cv::Range(rect.y, rect.y+rect.height), cv::Range(rect.x, rect.x+rect.height));
    float sum = 0;
    for(int i=rect.y; i<(rect.y+rect.height); i++){
        for(int j=rect.x; j<(rect.x+rect.width); j++){
            if(img.at<cv::Vec3b>(i,j)[0] != 0){
                sum = sum + 1;
            }
        }
    }
    sum = sum / (rect.width*rect.height);
    if(sum>0.2){
        return true;
    }
    return false;
}

void matSize(const cv::Mat& mat){
    std::cout << "Rows: " << mat.rows << " Cols: " << mat.cols << std::endl;
}

