#include "utils.h"

Ptr<TrainData> loadData(String csvPath){
    
    int headerLineCount = 1;
    int responseStartIdx = 0;
    
    Ptr<TrainData> trainingData = TrainData::loadFromCSV(csvPath, headerLineCount, responseStartIdx);
    
    return trainingData;
};

void displaySample(const Mat samples, int sampleNum) {
    // Reshape specific sample matrix to 28x28.
    Mat sampleImg = samples.row(sampleNum).reshape(1, 28);
    
    // Conver to single channel 8 bit image for display purposses.
    sampleImg.convertTo(sampleImg, CV_8UC1);
    
    imshow("digit", sampleImg);
    waitKey(0);
}