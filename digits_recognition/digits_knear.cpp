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

using namespace cv;
using namespace std;
using namespace ml;

int main(int argc, char* argv[] ) {

    String csv_traning_data = "../../../MNIST-data/train.csv";
    
    int headerLineCount = 1;
    int responseStartIdx = 0;
    int k = 3;

    // Load labeled training data from CSV file.
    Ptr<TrainData> trainingData = TrainData::loadFromCSV(csv_traning_data, headerLineCount, responseStartIdx);

    Mat true_labels = trainingData->getTrainResponses();
    Mat samples = trainingData->getTrainSamples();
    
    Mat trainLamples = true_labels(Range(0,35000), Range::all());
    Mat trainSamples = samples(Range(0,35000), Range::all());
    Mat testLabels = true_labels(Range(39000, 40000), Range::all());
    Mat testSamples = samples(Range(39000, 40000), Range::all());
    Mat predLabels = Mat::zeros(testLabels.size(), CV_32F);
    

    // Define our KNN classifier
    Ptr<KNearest> knn = KNearest::create();
    
    // Configure the KNN
    knn->setDefaultK(k);
    knn->setIsClassifier(true);
    
    // Train KNN (simply stores train data)
    knn->train(trainSamples, ROW_SAMPLE, trainLamples);
    
    // Predict using the labels of test data
    knn->findNearest(testSamples, k, predLabels);
    
    return 0;
}