#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;
using namespace ml;

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

int main(int argc, char* argv[] ) {
    
    Ptr<TrainData> trainingData = loadData("../../../MNIST-data/train.csv");
    
    Mat csvSamples = trainingData->getTrainSamples();
    
    // Go trough every sample digit in our dataset.
    for (int i=0; i < csvSamples.total(); ++i) {
        displaySample(csvSamples, i);
    }
    
    return 0;
}