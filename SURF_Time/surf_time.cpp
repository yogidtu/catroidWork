/*
*	A quick and simple test for testing SURF algorithm
*	We can change the computation time by changing
*		A) By minHis value - No of interest points.Larger the minHis value, smaller the no of interest points. 
*		B) Camera Resolution - I used (480*640). At high resolution the SURF algorithm has to surf a lot of pixels for each captured frame. 
*/

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>

#define minHis 750
#define matchingRatio 0.8

using namespace std;
using namespace cv;

int main()
{
	time_t start,musttime,end;
	Mat trainingImage1 ;
	trainingImage1= imread("BottleTrainingImage.jpg",CV_LOAD_IMAGE_GRAYSCALE);							

	if(!trainingImage1.data)
	{
		cout<<"No training Image\n";
		return -1;
	}
	else
	{
		cout<<"Training Image Loaded\n";
	}

	/*
	*	Detecting Key Points or Interest Points and save them
	*/

	SurfFeatureDetector detector(minHis);							
	std::vector<KeyPoint> kpTrainingImage1;

	start = clock();
	detector.detect(trainingImage1,kpTrainingImage1);	
	end = clock();
	cout << "Time for detecting key points of training image " << float( end -start)/CLOCKS_PER_SEC <<"seconds\n";

	/*
	*	Computing Descriptors and save them
	*/
	Mat desTrainingImage1;
	SurfDescriptorExtractor extractor;
	start = clock();
	extractor.compute(trainingImage1,kpTrainingImage1,desTrainingImage1);
	end = clock();
	cout << "Time for computing descriptors of training image " << float( end -start)/CLOCKS_PER_SEC <<"seconds\n\n\n";
	
	/*
	*	Corners of Training Image
	*/
	std::vector< Point2f> tiCorner(4),qiCorner(4);
	tiCorner[0] = cvPoint(0,0);
	tiCorner[1] = cvPoint( trainingImage1.cols, 0 );
	tiCorner[2] = cvPoint( trainingImage1.cols,trainingImage1.rows );
    tiCorner[3] = cvPoint( 0, trainingImage1.rows );


	//  Creating Window for Showing Results
	namedWindow("ShowingResult",CV_WINDOW_AUTOSIZE);
	
	/*
	*	Starting Camera 
	*/
	VideoCapture videoCapture(0);
	if (!videoCapture.isOpened())
	{
		return -1;
	}
	else 
	{
		cout<<" Camera Starts\n";
	}
	
	Mat frame;
	Mat queryImage;
	int myCount=0;
	int letItStart =0;

	while(1)
	{
	
	/*
	*	Capturing Frame
	*/
	start = clock();
	videoCapture >> frame;
	cvtColor(frame,queryImage,CV_RGB2GRAY,0);
	
	cout<< "Frame No\t"<<++myCount<<"\n";

	if (letItStart < 6)
       {
            letItStart++;
            continue;
        }
	

	/*
	*	Detecting Key points and
	*	computing descriptors 
	*	for each captured frame
	*/
	std::vector<KeyPoint> kpQueryImage;
	Mat desQueryImage;
	detector.detect(queryImage,kpQueryImage);
	extractor.compute(queryImage,kpQueryImage,desQueryImage);
	
	
	/*
	*	Matching Training Image descriptor with
	*	and captured frame(query image) descriptor
	*	using knn match
	*/
	std::vector<vector<DMatch> >initialMatches;
	FlannBasedMatcher matcher;
	matcher.knnMatch(desTrainingImage1,desQueryImage,initialMatches,2);

	/*
	*	Finding Good Matches and saving them
	*/
	vector<DMatch> goodMatches;
	for(int i=0; i < initialMatches.size(); i++)
	{
		if(  ((int) initialMatches[i].size() <=2) && ((int) initialMatches[i].size()>0) && (initialMatches[i][0].distance < matchingRatio*(initialMatches[i][1].distance)) )
		{
			//cout<<"pushing into googd matches\n";
			goodMatches.push_back(initialMatches[i][0]);
		}
	}

	end = clock();
	cout<<"Must Time" <<float(end - start)/CLOCKS_PER_SEC <<"\n";

	vector<Point2f> tiMatchPoint;
	vector<Point2f>	qiMatchPoint;
	Mat matchingImage;

	drawMatches( trainingImage1, kpTrainingImage1, queryImage, kpQueryImage, goodMatches, matchingImage, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	if( goodMatches.size() >= 25)
	{
		for (int i=0; i < goodMatches.size(); i++)
		{
			tiMatchPoint.push_back( kpTrainingImage1[goodMatches[i].queryIdx].pt);
			qiMatchPoint.push_back ( kpQueryImage[goodMatches[i].trainIdx].pt);
		}

	Mat homography = findHomography( tiMatchPoint, qiMatchPoint,CV_RANSAC);
	perspectiveTransform( tiCorner, qiCorner, homography);

	/*
	*	Drawing Lines across matching frame
	*/
    line( matchingImage, qiCorner[0] + Point2f( trainingImage1.cols, 0), qiCorner[1] + Point2f( trainingImage1.cols, 0), Scalar(0, 255, 0), 4 );
    line( matchingImage, qiCorner[1] + Point2f( trainingImage1.cols, 0), qiCorner[2] + Point2f( trainingImage1.cols, 0), Scalar( 0, 255, 0), 4 );
    line( matchingImage, qiCorner[2] + Point2f( trainingImage1.cols, 0), qiCorner[3] + Point2f( trainingImage1.cols, 0), Scalar( 0, 255, 0), 4 );
    line( matchingImage, qiCorner[3] + Point2f( trainingImage1.cols, 0), qiCorner[0] + Point2f( trainingImage1.cols, 0), Scalar( 0, 255, 0), 4 );
    
	cout<<"Match\n";
	}
	else
	{
		cout << "No Matches for this frame\n";
	}

	end = clock();
	cout << "Time for frame no "<<myCount<<"is " << float( end -start)/CLOCKS_PER_SEC <<"seconds\n\n\n";
	imshow( "ShowingResult", matchingImage);
	waitKey(1);
	
	}
	return 0;
}