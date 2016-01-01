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
using namespace cv;
using namespace std;

Mat extract_features(Mat img) {
    
    const int MIN_CONTOUR_AREA = 150;
    const int MAX_CONTOUR_AREA = 10000;
    
    Mat imgGray;
    Mat imgBin;
    Mat imgChar;
    Mat imgContours = Mat::zeros(img.size(), CV_8UC3);
    Scalar contourColor(255,255,255);
    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    
    // Conver to grayscale.
    cvtColor(img, imgGray, CV_BGR2GRAY);
    
    // Make tresholded binary image.
    GaussianBlur(imgGray, imgGray, Size(9, 9), 0);
    adaptiveThreshold(imgGray, imgBin, 255,
                      ADAPTIVE_THRESH_GAUSSIAN_C,
                      THRESH_BINARY_INV, 11, 0);
    
    // Extract contourse from binary image.
    findContours(imgBin, contours,
                 hierarchy, RETR_EXTERNAL,
                 CHAIN_APPROX_SIMPLE);
    
    for (int i = 0; i < contours.size(); i++) {
        drawContours(imgContours, contours, i,
                     contourColor, 0.1, 8, hierarchy,
                     0, Point());
    }
    
    return imgContours;
};

int main(int argc, char* argv[] )
{
    
    Mat img = imread("/Users/jacobrafati/Desktop/DigitRec-openCV/digits_recognition/test_data/numbers_01.jpg");
    Mat outImg = extract_features(img);
    
    imshow("outWindow", outImg);
    waitKey();
    
    //    namedWindow("image", WINDOW_AUTOSIZE);
    //    namedWindow("char", WINDOW_AUTOSIZE);
    
    //    Mat img;
    //    Mat matThresh;
    //    Mat charRoi;
    //    vector<Vec4i> hierarchy;
    //    vector<vector<Point> > contours;
    
    //    VideoCapture cap;
    //    cap.open(0);
    
    //    while (1) {
    //        cap >> img;
    
    //        cvtColor(img, matThresh, CV_BGR2GRAY);
    //        GaussianBlur(matThresh, matThresh, Size(9, 9), 0);
    //        adaptiveThreshold(matThresh, matThresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 0);
    //
    //        findContours(matThresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //
    //        Mat drawing = Mat::zeros( matThresh.size(), CV_8UC3 );
    //        Scalar color( 255,255,255);
    //        for (int i = 0; i < contours.size(); i++) {
    //            if (contourArea(contours[i]) > MIN_CONTOUR_AREA && contourArea(contours[i]) < MAX_CONTOUR_AREA) {
    //                Rect boundingRect = cv::boundingRect(contours[i]);
    //                rectangle(drawing, boundingRect, Scalar(0, 0, 255), 2);
    //
    //                drawContours( drawing, contours, i, color, 0.1, 8, hierarchy, 0, Point() );
    //
    //                charRoi = img(boundingRect);
    //
    //                imshow("drawing", drawing);
    //                waitKey(0);
    //
    //                resize(charRoi, charRoi, Size(100, 100));
    //
    //                cout << "Rows: " << charRoi.rows << endl;
    //                cout << "Cols: " << charRoi.cols << endl;
    //
    //                imshow("charRoi", charRoi);
    //                moveWindow("charRoi", 0, 300);
    //                
    //                waitKey(0);
    //            }
    //            
    //        }
    //    }
    
    return 0;
}
