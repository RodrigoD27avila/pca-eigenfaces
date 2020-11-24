#include <iostream>
#include <regex>
#include <filesystem>
#include <random>
#include <limits>
#include <iomanip>

#include <vector>
#include <array>

#include "Person.h"
#include "PCAEigenFace.h"

std::mt19937_64 random_engine{42};

Mat getImageData(std::string fileName)
{
    Mat img = imread(fileName, IMREAD_GRAYSCALE);

    // Muda o tamanho para 80x80
    Mat dst;
    resize(img, dst, Size(80, 80));

    //Converter para vetor coluna
    // 1 2
    // 3 4
    //1 3
    //2 4
    // 1
    // 3
    // 2
    // 4
    dst = dst.t();
    dst = dst.reshape(1, dst.cols * dst.rows);

    // Converte de 8 bits sem sinal, para 64 bits com sinal, preserva 1 canal apenas.
    Mat data;
    dst.convertTo(data, CV_64FC1);

    return data;
}

Person toPerson(std::string fileName)
{
    Person person{};

    std::regex re("(.*)/(\\d+)_(\\d+)(.*)");
    std::smatch match;

    if (std::regex_search(fileName, match, re) == true)
    {
        person.id = std::stoi(match.str(2));
        person.label = std::stoi(match.str(3));
        person.data = getImageData(fileName);
    }

    return person;
}

void loadDataset(const std::string &path, std::vector<Person> &train, std::vector<Person> &test, float p)
{

    std::vector<Person> people;
    for (const auto &entry : std::filesystem::directory_iterator(path))
    {
        Person person = toPerson(entry.path());
        people.push_back(person);
    }

    std::shuffle(people.begin(), people.end(), random_engine);

    size_t offset = int(p * people.size());

    std::copy(people.begin(), people.begin() + offset, std::back_inserter(train));
    std::copy(people.begin() + offset, people.end(), std::back_inserter(test));
}

int main()
{
    std::vector<Person> train;
    std::vector<Person> test;
    float p = 0.7;

    std::string path = "ORL2";
    loadDataset(path, train, test, p);

    double minDistance = std::numeric_limits<double>::max();
    double maxDistance = std::numeric_limits<double>::min();
    double meanDistance = 0;
    int corrects = 0;

    double minRec = std::numeric_limits<double>::max();
    double maxRec = std::numeric_limits<double>::min();
    double meanRec = 0;

    const double MAX_DISTANCE = 2500;
    const double MAX_REC = 2900;

    for (int numComponents : {10, 15, 20})
    {
        PCAEigenFace model{numComponents};
        model.train(train);

        int truePositiveCount = 0;
        int trueNegativesCount = 0;

        for (Person &personToTest : test)
        {
            Mat testData = personToTest.data;
            std::array<int, 1> label{};
            std::array<double, 1> confidence{};
            std::array<double, 1> reconstructionError{};

            model.predict(testData, label, confidence, reconstructionError);

            bool labelOK = label[0] == personToTest.label;
            if (labelOK)
            {
                corrects++;
            }

            if (reconstructionError[0] > MAX_REC)
            {

                if (!labelOK)
                {
                    trueNegativesCount++;
                }
            }
            else if (confidence[0] > MAX_DISTANCE)
            {

                if (!labelOK)
                {
                    trueNegativesCount++;
                }
            }
            else if (reconstructionError[0] > 2400 && confidence[0] > 1800)
            {

                if (!labelOK)
                {
                    trueNegativesCount++;
                }
            }
            else if (labelOK)
            {
                truePositiveCount++;
            }

            if (personToTest.label <= 40)
            {
                // definir um limiar de confiança/distância de confiança
                if (confidence[0] < minDistance)
                {
                    minDistance = confidence[0];
                }

                if (confidence[0] > maxDistance)
                {
                    maxDistance = confidence[0];
                }

                meanDistance += confidence[0];

                // definir um limiar de confiança/distância de confiança
                if (reconstructionError[0] < minRec)
                {
                    minRec = reconstructionError[0];
                }

                if (reconstructionError[0] > maxRec)
                {
                    maxRec = reconstructionError[0];
                }

                meanRec += reconstructionError[0];
            }
        } //for

        int trues = truePositiveCount + trueNegativesCount;

        double accuracy = double(trues) / double(test.size()) * 100.0;

        std::cout << numComponents << " componentes principais, acurácia: "
                  << std::setprecision(2) << accuracy << "%." << std::endl;
    }

    return 0;
}