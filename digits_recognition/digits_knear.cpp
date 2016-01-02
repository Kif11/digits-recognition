//
//  main.cpp
//  digits_recognition
//
//  Created by Amy on 12/31/15.
//  Copyright (c) 2015 Kirill. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/ml.hpp>

#include "utils.h"

using namespace cv;
using namespace std;
using namespace ml;

int main(int argc, char* argv[] ) {
    
    int k = 5;
    int totalSamples = 42000;
    int totalTrain = totalSamples * 0.99;
    Range trainRows = Range(0, totalTrain);
    Range testRows = Range(totalTrain, totalSamples);
    Range allColumns = Range::all();

    // Load labeled training data from CSV file.
    Ptr<TrainData> trainingData = loadData("../../../MNIST-data/train.csv");
    
    // Raw matrices from  CSV file.
    Mat trueLabels = trainingData->getTrainResponses();
    Mat samples = trainingData->getTrainSamples();
    
    // Train matrices.
    Mat trainLabels = trueLabels(trainRows, allColumns);
    Mat trainSamples = samples(trainRows, allColumns);
    
    // Test matrices.
    Mat testLabels = trueLabels(testRows, allColumns);
    Mat testSamples = samples(testRows, allColumns);
    
    // Matrix for storing prediction labels.
    Mat predLabels = Mat::zeros(testLabels.size(), CV_8U);
    
    // Define our KNN classifier
    Ptr<KNearest> knn = KNearest::create();
    
    // Configure the KNN
    knn->setDefaultK(k);
    knn->setIsClassifier(true);
    
    // Train KNN (simply stores train data)
    knn->train(trainSamples, ROW_SAMPLE, trainLabels);
    
    // Predict using the labels of test data
    knn->findNearest(testSamples, k, predLabels);
    
    int totalCorrect = 0;
    // For each predicted label.
    for (int i = 0; i < predLabels.rows; ++i) {
        float* testLb = testLabels.ptr<float>(0, i);
        float* predLb = predLabels.ptr<float>(0, i);
        if (*testLb == *predLb) {
            totalCorrect++;
        }
//        else {
//            cout << "Predicted label: " << *predLb << endl;
//            displaySample (testSamples, i);
//        }
    }
    cout << "Total correct " << totalCorrect << " out of " << testSamples.rows << endl;
    
    return 0;
}