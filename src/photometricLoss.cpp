
// read image
// select pixel
// optimization
// pyramid improvement

//local header files
#include <reprojection.h>
#include <photometricBA.h>
#include "ultils.h"
#include "PCLOpt.h"
#include "dataLoader.h"

//#include <algorithm>
//#include <atomic>
//#include <chrono>

#include <sophus/se3.hpp>
//#include <tbb/concurrent_unordered_map.h>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
//#include "opencv2/features2d.hpp"



#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>
#include <visnav/local_parameterization_se3.hpp>

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
	depth_target=dataLoader->depth_map_target;
	image_ref_baseColor= dataLoader->image_ref_baseColor;

	Eigen::Matrix3d K;
	K=dataLoader->camera_intrinsics;
	double fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy = K(1, 2), f=30.0;

	Sophus::SE3d xi, xi_GT;
	Eigen::Matrix<double, 3,3> R;
	R=dataLoader->q_12 .normalized().toRotationMatrix();

	// initialize the pose xi
//	xi.setRotationMatrix(R);
//	xi.translation()=dataLoader->t12;

	xi_GT.setRotationMatrix(R);
	xi_GT.translation()=dataLoader->t12;


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

			Eigen::Vector3d normal_new( normal_map_GT.at<Vec3f>(u,v)[2],  -normal_map_GT.at<Vec3f>(u,v)[1], normal_map_GT.at<Vec3f>(u,v)[0]);// !!!!!!!!!!!!!!!!!!!!!!!

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

	int lvl_target, lvl_ref;
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
		Mat deltaMap(depth_ref.rows, depth_ref.cols, CV_32FC1, Scalar(1)); // storing delta
        int i=0;
		while ( i < 2){

			PhotometricBA(IRef, I, options, Klvl, xi, DRef,deltaMap);
			updateDelta(xi,Klvl,image_ref_baseColor,DRef,image_ref_metallic ,image_ref_roughness,light_source,deltaMap,newNormalMap,up_new, butt_new);

//			Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K,distanceThres,xi_GT, upper, buttom, deltaMap);
//			Mat showGTdeltaMap=colorMap(deltaMapGT_res, upper, buttom);
//			Mat showESdeltaMap=colorMap(deltaMap, upper, buttom);
//			imshow("show GT deltaMap", showGTdeltaMap);
//			imshow("show ES deltaMap", showESdeltaMap);
//			imwrite("GT_deltaMap.exr",showGTdeltaMap);
//			imwrite("ES_deltaMap.exr",showESdeltaMap);
//
			cout << "\n Show initial pose:\n" << xi_GT.rotationMatrix() << "\n Show translation:\n" << xi_GT.translation()<<endl;
			cout << "\n Show optimized pose:\n" << xi.rotationMatrix() << "\n Show translation:\n" << xi.translation()<< endl;
			cout << "\n Show Rotational error :"<< rotationErr(xi_GT.rotationMatrix(), xi.rotationMatrix()) <<"(degree)."<<"\n Show translational error :" << 100* translationErr(xi_GT.translation(), xi.translation()) <<"(%) "<<endl;
//			waitKey(0);




          i+=1;

		}



		cout << "\n Show optimized pose:\n" << xi.rotationMatrix() << "\n Show translation:\n" << xi.translation()
		     << endl;
		Eigen::Quaterniond q_opt( xi.rotationMatrix());
		cout<<"\n Show the optimized rotation as quaternion:"<<q_opt.w()<<","<<q_opt.x()<<","<<q_opt.y()<<","<<q_opt.z()<< endl;

	}
	// tidy up
	delete dataLoader;
	return 0;
}
