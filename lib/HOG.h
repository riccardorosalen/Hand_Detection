#ifndef HOG_H_INCLUDED
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>

class HOGwSVM {
private:
	cv::HOGDescriptor hog;
public:
	//The Following functions are here only to show what was used for training the detector
	//It's a simple edit of the code in "https://docs.opencv.org/3.4/d0/df8/samples_2cpp_2train_HOG_8cpp-example.html#a33"
	//The whole original code can be find at the link
	std::vector< float > get_svm_detector(const cv::Ptr< cv::ml::SVM >& svm);
	void convert_to_ml(const std::vector< cv::Mat >& train_samples, cv::Mat& trainData);
	void load_images(const std::string& dirname, std::vector< cv::Mat >& img_lst, bool showImages);
	void sample_neg(const std::vector< cv::Mat >& full_neg_lst, std::vector< cv::Mat >& neg_lst, const cv::Size& size);
	void computeHOGs(const cv::Size wsize, const std::vector< cv::Mat >& img_lst, std::vector< cv::Mat >& gradient_lst, bool use_flip);
	void train_new_detector(std::string test_dir, std::string pos_dir, std::string neg_dir, std::string obj_det_filename);
	void test_trained_detector(std::string obj_det_filename, std::string test_dir);

	//Computes HOG::detectMultiScale for the image passed as argument
	std::vector<cv::Rect> detectRect(cv::Mat img);
	//Given the name of the file loads an HOGDescriptor file with a pre-trained SVM detector
	void load_trained_detector(std::string detector_filename);

};

#endif
