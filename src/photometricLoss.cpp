
// read image
// select pixel
// optimization
// pyramid improvement

//local header files
#include "reprojection.h"
#include "photometricBA.h"
#include "ultils.h"
#include "PCLOpt.h"
#include "dataLoader.h"

//#include <algorithm>
//#include <atomic>
//#include <chrono>

#include <sophus/se3.hpp>
//#include <tbb/concurrent_unordered_map.h>
#include <unordered_map>
//#include <Eigen/Dense>
#include <Eigen/Core>
//#include <Eigen/Geometry>

//#include <Eigen/StdVector>
//#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
//#include "opencv2/features2d.hpp"



//#include <ceres/ceres.h>
//#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>
//#include <visnav/local_parameterization_se3.hpp>

using namespace cv;
using namespace std;
using namespace DSONL;


int main(int argc, char **argv) {


	Eigen::Vector2i pixel_pos(213,295);
	std::unordered_map<int, int> inliers_filter;
	inliers_filter.emplace(173,333); //yes
	inliers_filter.emplace(378,268); //yes

	// data loader
	Mat grayImage_target, grayImage_ref,depth_ref,depth_target,image_ref_baseColor;
	dataLoader* dataLoader;
	dataLoader= new DSONL::dataLoader();
	dataLoader->Init();

	Mat image_ref_metallic =dataLoader->image_ref_metallic;
	Mat image_ref_roughness=dataLoader->image_ref_roughness;

	grayImage_ref=dataLoader->grayImage_ref;
	grayImage_target=dataLoader->grayImage_target;

	grayImage_ref.convertTo(grayImage_ref,CV_64FC1);
	grayImage_target.convertTo(grayImage_target, CV_64FC1);

//	showImage(grayImage_ref,"grayImage_ref");
//	showImage(grayImage_target,"grayImage_target");
//	waitKey(0);

	depth_ref=dataLoader->depth_map_ref;
	// set all nan zero
	Mat depth_mask = Mat(depth_ref != depth_ref);
	depth_ref.setTo(0.0, depth_mask);
	depth_ref.convertTo(depth_ref, CV_64FC1);
	// Add noise to depth image depth_ref_NS
	Mat depth_ref_NS;
	Mat depth_ref_GT= dataLoader->depth_map_ref;
	double Mean=0.0,StdDev=5.0;
	AddGaussianNoise_Opencv(depth_ref_GT,depth_ref_NS,Mean,StdDev);
	depth_ref_NS.convertTo(depth_ref_NS, CV_64FC1);


	// show the depth image with noise
		double min_gt, max_gt;
		cv::minMaxLoc(depth_ref, &min_gt, &max_gt);
		cout<<"\n show original depth_ref min, max:\n"<<min_gt<<","<<max_gt<<endl;
//		Mat depth_ref_show= depth_ref*(1.0/(max_gt-min_gt))+(-min_gt*(1.0/(max_gt-min_gt)));
	//	cv::minMaxLoc(depth_ref_NS, &min_gt, &max_gt);
	//	cout<<"\n show depth_ref_NS min, max:\n"<<min_gt<<","<<max_gt<<endl;
	//	Mat depth_ref_NS_show= depth_ref_NS*(1.0/(max_gt-min_gt))+(-min_gt*(1.0/(max_gt-min_gt)));
	//
	//	// show depth image and noise one
//		depth_ref_show.convertTo(depth_ref_show, CV_32FC1);
//		imshow("depth_ref",depth_ref_show);
	//	imshow("depth_ref_NS",depth_ref_NS_show);
	//	waitKey(0);


	depth_target=dataLoader->depth_map_target;
	image_ref_baseColor= dataLoader->image_ref_baseColor;

	Eigen::Matrix3d K;
	K=dataLoader->camera_intrinsics;
	double fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy = K(1, 2), f=30.0;

	// ----------------------------------------optimization variable: R, t--------------------------------------
	Sophus::SE3d xi, xi_GT;
	Eigen::Matrix<double, 3,3> R;
	R=dataLoader->q_12 .normalized().toRotationMatrix();
	// initialize the pose xi
	//	xi.setRotationMatrix(R);
	//	xi.translation()=dataLoader->t12;

	xi_GT.setRotationMatrix(R);
	xi_GT.translation()=dataLoader->t12;
	// ----------------------------------------optimization variable: depth --------------------------------------




	cout << "\n Show initial pose:\n" << xi.rotationMatrix() << "\n Show translation:\n" << xi.translation()<<endl;
	cout << "\n Show GT pose:\n" << xi_GT.rotationMatrix() << "\n Show GT translation:\n" << xi_GT.translation()<<endl;

// ------------------------------------------------------------------------------------------Movingleast algorithm---------------------------------------------------------------
	std::vector<Eigen::Vector3d> pts;
	cv::Mat normal_map(depth_ref.rows, depth_ref.cols, CV_64FC3);
	//MLS();
//	---------------------------------------------------------normal_map_GT---------------------------------------------------
	Mat normal_map_GT;
	normal_map_GT=dataLoader->normal_map_GT;
	for (int u = 0; u< depth_ref.rows; u++) // colId, cols: 0 to 480
	{
		for (int v = 0; v < depth_ref.cols; v++) // rowId,  rows: 0 to 640
		{

			Eigen::Vector3d normal_new( normal_map_GT.at<Vec3f>(u,v)[2],  -normal_map_GT.at<Vec3f>(u,v)[1], normal_map_GT.at<Vec3f>(u,v)[0]);
			normal_new= dataLoader->R1.transpose()*normal_new;

			Eigen::Vector3d principal_axis(0, 0, 1);
			if(normal_new.dot(principal_axis)>0)
			{
				normal_new = -normal_new;
			}

			normal_map.at<Vec3d>(u,v)[0]=normal_new(0);
			normal_map.at<Vec3d>(u,v)[1]=normal_new(1);
			normal_map.at<Vec3d>(u,v)[2]=normal_new(2);

		}
	}


	Mat newNormalMap=normal_map;
	double distanceThres=0.07;
	float upper=5;
	float buttom=0.2;
	float up_new=upper;
	float butt_new=buttom;
	Mat deltaMap(depth_ref.rows, depth_ref.cols, CV_32FC1, Scalar(1)); // storing delta
	int lvl_target, lvl_ref;
	double depth_upper_bound =60;
	double depth_lower_bound =20;


	for (int lvl = 1; lvl >= 1; lvl--)
	{
		cout << "\n Show the value of lvl:" << lvl << endl;
		Mat IRef, DRef, I, D;
		Eigen::Matrix3d Klvl, Klvl_ignore;
		lvl_target = lvl;
		lvl_ref = lvl;

		downscale(grayImage_ref, depth_ref, K, lvl_ref, IRef, DRef, Klvl);
		downscale(grayImage_target, depth_target, K, lvl_target, I, D, Klvl_ignore);

		PhotometricBAOptions options;
		options.optimize_depth= true;

        int i=0;
		Mat depth_ref_show_before= depth_ref*(1.0/(max_gt-min_gt))+(-min_gt*(1.0/(max_gt-min_gt)));
		while ( i < 2){
//			Mat showESdeltaMap=colorMap(deltaMap, upper, buttom);
//			imshow("show ES deltaMap", showESdeltaMap);
			double  max_n_, min_n_;
			cv::minMaxLoc(deltaMap, &min_n_, &max_n_);
			cout<<"->>>>>>>>>>>>>>>>>show max and min of estimated deltaMap:"<< max_n_ <<","<<min_n_<<endl;
			Mat mask = cv::Mat(deltaMap != deltaMap);
			deltaMap.setTo(1.0, mask);

//			Mat depth_ref_show_nside_while= depth_ref*(1.0/(max_gt-min_gt))+(-min_gt*(1.0/(max_gt-min_gt)));
//			imshow("inside_while", depth_ref_show_nside_while);
//			cout<<"1 show depth of depth_ref : "<<depth_ref.depth()<<endl;

			double min_gt_special, max_gt_special;
			cv::minMaxLoc(depth_ref, &min_gt_special, &max_gt_special);
			cout<<"\n show depth_ref min, max:\n"<<min_gt_special<<","<<max_gt_special<<endl;
			Mat depth_ref_for_show= depth_ref*(1.0/(max_gt_special-min_gt_special))+(-min_gt_special*(1.0/(max_gt_special-min_gt_special)));


			string depth_ref_name= "depth_ref"+ to_string(i);
			imshow(depth_ref_name, depth_ref_for_show);


			PhotometricBA(IRef, I, options, Klvl, xi, depth_ref,deltaMap,depth_upper_bound, depth_lower_bound);
			updateDelta(xi,Klvl,image_ref_baseColor,depth_ref,image_ref_metallic ,image_ref_roughness,light_source, deltaMap,newNormalMap,up_new, butt_new);

//			Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K,distanceThres,xi_GT, upper, buttom, deltaMap);
//			Mat showGTdeltaMap=colorMap(deltaMapGT_res, upper, buttom);
//			Mat showESdeltaMap=colorMap(deltaMap, upper, buttom);
//			imshow("show GT deltaMap", showGTdeltaMap);
//			imshow("show ES deltaMap", showESdeltaMap);
//			imwrite("GT_deltaMap.exr",showGTdeltaMap);
//			imwrite("ES_deltaMap.exr",showESdeltaMap);
//
			cout<<"\n show depth_ref min, max:\n"<<min_gt_special<<","<<max_gt_special<<endl;
			cout << "\n Show initial pose:\n" << xi_GT.rotationMatrix() << "\n Show translation:\n" << xi_GT.translation()<<endl;
			cout << "\n Show optimized pose:\n" << xi.rotationMatrix() << "\n Show translation:\n" << xi.translation()<< endl;
			cout << "\n Show Rotational error :"<< rotationErr(xi_GT.rotationMatrix(), xi.rotationMatrix()) <<"(degree)."<<"\n Show translational error :" << 100* translationErr(xi_GT.translation(), xi.translation()) <<"(%) "<<endl;
//			waitKey(0);
          i+=1;

		}
		waitKey(0);


//		cout << "\n Show initial pose:\n" << xi_GT.rotationMatrix() << "\n Show translation:\n" << xi_GT.translation()<<endl;
//		cout << "\n Show optimized pose:\n" << xi.rotationMatrix() << "\n Show translation:\n" << xi.translation()<< endl;
//		cout << "\n Show Rotational error :"<< rotationErr(xi_GT.rotationMatrix(), xi.rotationMatrix()) <<"(degree)."<<"\n Show translational error :" << 100* translationErr(xi_GT.translation(), xi.translation()) <<"(%) "<<endl;

	}
	// tidy up
	delete dataLoader;
	return 0;
}
