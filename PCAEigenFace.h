#ifndef __PCAEIGENFACE_H_INCLUDED__
#define __PCAEIGENFACE_H_INCLUDED__

#include <vector>
#include <opencv2/opencv.hpp>

#include "Person.h"

using namespace cv;

class PCAEigenFace {
public:
    PCAEigenFace(int numComponents) : numComponents(numComponents)
    {}

    void train(const std::vector<Person>& train);

    void predict(Mat testData, std::array<int, 1>& label, std::array<double, 1>& confidence, std::array<double, 1>& reconstructionError);
private:
    
    void calcProjections(const std::vector<Person>&  train);

    void calcEigenFaces();

    void calcEigen();

    void calcCovariance();

    Mat mul(Mat a, Mat b);

    void calcDiff(const std::vector<Person>& train);

    void calcMean(const std::vector<Person>& train);

    Mat calcReconstruction(Mat w);

    double calcDistance(Mat p, Mat q);

private:
	int numComponents;
	Mat mean;
	Mat diffs;
	Mat covariance;
	Mat eigenvectors;
	Mat eigenvalues;
	Mat eigenFaces;
	std::vector<int> labels;
	Mat projections;
};

#endif // __PCAEIGENFACE_H_INCLUDED__
