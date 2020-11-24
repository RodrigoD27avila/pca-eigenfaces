#include "PCAEigenFace.h"

void PCAEigenFace::train(const std::vector<Person> &train)
{
    calcMean(train);
    calcDiff(train);
    calcCovariance();
    calcEigen();
    calcEigenFaces();
    calcProjections(train);
}

void PCAEigenFace::predict(Mat testData, std::array<int, 1> &label, std::array<double, 1> &confidence, std::array<double, 1> &reconstructionError)
{
    Mat diff;
    subtract(testData, mean, diff);

    // Calcula os pesos da imagem desconhecida.
    Mat w = mul(eigenFaces.t(), diff);

    // Calcular o vizinho mais próximo dessa projeção "desconhecida".
    int minJ = 0;
    double minDistance = calcDistance(w, projections.col(minJ));
    for (int j = 1; j < projections.cols; j++)
    {
        double distance = calcDistance(w, projections.col(j));
        if (distance < minDistance)
        {
            minDistance = distance;
            minJ = j;
        }
    }

    label[0] = labels[minJ];
    confidence[0] = minDistance;

    Mat reconstruction = calcReconstruction(w);
    reconstructionError[0] = norm(testData, reconstruction, NORM_L2);
}

void PCAEigenFace::calcProjections(const std::vector<Person> &train)
{
    labels.reserve(train.size());
    projections = Mat(numComponents, train.size(), CV_64FC1);
    for (int j = 0; j < diffs.cols; j++)
    {
        Mat diff = diffs.col(j);
        Mat w = mul(eigenFaces.t(), diff);
        w.copyTo(projections.col(j));
        labels[j] = train[j].label;
    }
}

void PCAEigenFace::calcEigenFaces()
{
    // Transposição dos autovetores.
    // 1 2 3
    // 4 5 6
    // 1 4
    // 2 5
    // 3 6
    Mat evt = eigenvectors.t();
    Mat ev_k = evt.colRange(0, numComponents > 0 ? numComponents : evt.cols);
    for (int j = 0; j < ev_k.cols; j++)
    {
        evt.col(j).copyTo(ev_k.col(j));
    }

    eigenFaces = mul(diffs, ev_k);
    for (int j = 0; j < eigenFaces.cols; j++)
    {
        Mat ef = eigenFaces.col(j);
        // Normalização L2 = Yi = Xi / sqrt(sum((Xi)^2)), onde i = 0...rows-1
        normalize(ef, ef);
    }
}

void PCAEigenFace::calcEigen()
{
    eigen(covariance, eigenvalues, eigenvectors);
}

void PCAEigenFace::calcCovariance()
{
    covariance = mul(diffs.t(), diffs);
}

Mat PCAEigenFace::mul(Mat a, Mat b)
{
    Mat c(a.rows, b.cols, CV_64FC1);
    gemm(a, b, 1, Mat(), 1, c);
    return c;
}

void PCAEigenFace::calcDiff(const std::vector<Person> &train)
{
    Mat sample = train[0].data;
    diffs = Mat(sample.rows, train.size(), sample.type());
    for (int i = 0; i < diffs.rows; i++)
    {
        for (int j = 0; j < diffs.cols; j++)
        {
            double mv = mean.at<double>(i, 0);
            Mat data = train[j].data;
            double dv = data.at<double>(i, 0);
            double v = dv - mv;
            diffs.at<double>(i, j) = v;
        }
    }
}

void PCAEigenFace::calcMean(const std::vector<Person> &train)
{
    Mat sample = train[0].data;
    mean = Mat::zeros(sample.rows, sample.cols, sample.type());

    std::for_each(train.begin(), train.end(), [&](Person person) {
        Mat data = person.data;
        for (int i = 0; i < mean.rows; i++)
        {
            double mv = mean.at<double>(i, 0);
            double pv = data.at<double>(i, 0);
            mv += pv;
            mean.at<double>(i, 0) = mv;
        }
    });

    for (int i = 0; i < mean.rows; i++)
    {
        double mv = mean.at<double>(i, 0);
        mv /= train.size();
        mean.at<double>(i, 0) = mv;
    }
}

Mat PCAEigenFace::calcReconstruction(Mat w)
{
    Mat result = mul(eigenFaces, w);
    //result += mean;
    add(result, mean, result);

    return result;
}

double PCAEigenFace::calcDistance(Mat p, Mat q)
{
    // Distância euclidiana.
    // d = sqrt(sum(pi - qi)^2)
    double distance = 0;
    for (int i = 0; i < p.rows; i++)
    {
        double pi = p.at<double>(i, 0);
        double qi = q.at<double>(i, 0);
        double d = pi - qi;
        distance += d * d;
    }

    double result = std::sqrt(distance);

    return result;
}