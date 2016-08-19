#include "stdafx.h"
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv\highgui.h>

using namespace cv;
using namespace std;

String banana_cascade_name = "banana_classifier.xml";

CascadeClassifier banana_cascade;
CascadeClassifier spot_cascade;
string window_name = "Detection Window";
RNG rng(12345);

void detectAndDisplay(Mat frame);

int main(int argc, const char** argv)
{

	Mat aframe;
	VideoCapture cap;          //initialize capture
	cap.open(1);
	printf("Legume Prospector Loaded.\n");
	while (true)
	{
		//copy webcam stream to image
		cap >> aframe;          
		//delay 33ms
		waitKey(33);          
		detectAndDisplay(aframe);
		int c = cvWaitKey(10);
		if ((char)c == 27) { exit(0); }
	}
	return 0;
}

void detectAndDisplay(Mat frame)
{
	std::vector<Rect> bananas;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect bananas

	if (!banana_cascade.load(banana_cascade_name))
	{ printf("--(!)Error loading\n"); }


	banana_cascade.detectMultiScale(frame_gray, bananas, 2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 10));

	String infotext = "Banana Detected!";

	for (size_t i = 0; i < bananas.size(); i++)
	{
		
		Point center(bananas[i].x + bananas[i].width*0.5, bananas[i].y + bananas[i].height*0.5);
		ellipse(frame, center, Size(bananas[i].width*0.5, bananas[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		
		putText(frame, infotext, center, CV_FONT_NORMAL,1,CV_RGB(255,255,255));
		
		Mat faceROI = frame_gray(bananas[i]);
		std::vector<Rect> eyes;

		//-- In each banana, detect spots
		spot_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point center(bananas[i].x + eyes[j].x + eyes[j].width*0.5, bananas[i].y + eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
	}
	//-- Show what you got
	imshow(window_name, frame);
}