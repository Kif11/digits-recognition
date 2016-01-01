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
    // Load labeled training data (feature matrix) and label vector
    String filename = "/Users/jacobrafati/Desktop/DigitRec-openCV/digits_recognition/MNIST-data/train.csv";
    int headerLineCount = 1;
    int responseStartIdx = 0;

    Ptr<TrainData> trainingData = TrainData::loadFromCSV(filename, headerLineCount, responseStartIdx);
    
    // delete these lines if you are not going to use label vector and feature matrix
    Mat true_labels = trainingData->getTrainResponses();
    Mat samples = trainingData->getTrainSamples();
    
    
    Mat train_labels = true_labels(Range(0,35000), Range::all());
    Mat train_samples = samples(Range(0,35000), Range::all());
    
    Mat test_labels = true_labels(Range(35000, 40000) , Range::all());
    Mat test_samples = samples(Range(35000, 40000) , Range::all());
    
    
    Mat pred_labels;

    // define a classifier
    Ptr<KNearest> kclassifier = KNearest::create();
    
    // train KNN (simply stores train data)
    kclassifier->train(train_labels, ROW_SAMPLE, train_samples);
    
    int k = 3;
    
    
    // predict using the labels of test data
    kclassifier->findNearest(test_samples, k, pred_labels);

    
    return 0;
}