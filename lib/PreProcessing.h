#ifndef PREPROCESSING_H_INCLUDED

#include<opencv2/highgui.hpp>

//Applies closing operation to the image
cv::Mat morph(cv::Mat img);
//Applies several closing operations and generates a mask for the image
cv::Mat morphMask(cv::Mat img);
//Segments the skin inside the image setting all other pixels in black color
cv::Mat skinSeg(cv::Mat in);

#endif
