#ifndef __PERSON_H_INCLUDED__
#define __PERSON_H_INCLUDED__

#include <opencv2/opencv.hpp>

using namespace cv;

struct Person {
    int id;
    int label;
    Mat data;
};

#endif // __PERSON_H_INCLUDED__