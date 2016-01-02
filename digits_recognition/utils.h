//
//  utils.h
//  digits_recognition
//
//  Created by Amy on 1/1/16.
//  Copyright (c) 2016 Kirill. All rights reserved.
//

#ifndef digits_recognition_utils_h
#define digits_recognition_utils_h

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;
using namespace ml;

Ptr<TrainData> loadData(String csvPath);

void displaySample(const Mat samples, int sampleNum);

#endif
