//
// Created by cheng on 10.10.22.
//

#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>

#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <iostream>


namespace DSONL{
	using namespace cv;
	using namespace std;




	Eigen::Matrix<double,3,1> light_source(7.1, -22.9 ,11.0);
	Eigen::Matrix<double,3,1> Camera1(3.8, -16.5 ,26.1);
	Eigen::Matrix<double,3,1> Camera2(0.3 ,-16.9, 27.7);

	Eigen::Quaterniond q_1(0.009445649,-0.0003128,-0.9994076,-0.0330920); //  cam1  wxyz
	Eigen::Quaterniond q_2(-0.08078633,-0.0084485,-0.9962677,-0.0292077 ); //  cam5  wxyz
	Eigen::Vector3d t1( 3.8, -16.5, 26.1);
	Eigen::Vector3d t2(-5.1,-15.2 ,27.5);
	Eigen::Matrix3d R1=q_1.toRotationMatrix();
	Eigen::Matrix3d R2=q_2.toRotationMatrix();
	Eigen::Matrix3d R12= R2.transpose() * R1;
	Eigen::Vector3d t12= R2.transpose()* (t1-t2);
	Eigen::Quaterniond q_12(R12);
	Eigen::Vector3d l_w(0.223529, 0.490196, 0.843137);
	Eigen::Vector3d N_w(0.0352942, -0.223529, 0.976471);


	// RGB image with texture
	string image_ref_path = "../data/rgb/Texture_Image/rt_17_3_40_cam1_texture.exr";
	string image_target_path = "../data/rgb/Texture_Image/rt_17_3_40_cam5_texture.exr";

//	// HD RGB image with texture
//	string image_ref_path = "../data/rgb/HDdataset/rt_16_40_56_cam11__rgb.exr";
//	string image_target_path = "../data/rgb/HDdataset/rt_16_40_56_cam55__rgb.exr";

	// RGB image without texture
//	string image_ref_path = "../data/rgb/No_Texture_Images/rt_16_4_56_cam1_notexture.exr";
//	string image_target_path = "../data/rgb/No_Texture_Images/rt_16_4_56_cam5_notexture.exr";



//	// BaseColor Image with texture
	string image_ref_baseColor_path = "../data/rgb/Texture_Image/rt_17_4_52_cam1_texture_basecolor.exr";
	string image_target_baseColor = "../data/rgb/Texture_Image/rt_17_4_52_cam5_texture_basecolor.exr";

   // HD BaseColor Image with texture
//	string image_ref_baseColor_path = "../data/rgb/HDdataset/rt_16_35_53_cam11__basecolor.exr";
//	string image_target_baseColor = "../data/rgb/HDdataset/rt_16_35_53_cam55__basecolor.exr";

	// BaseColor Image without texture
//	string image_ref_baseColor_path = "../data/rgb/No_Texture_Images/rt_16_5_47_cam1_notexture_basecolor.exr";
//	string image_target_baseColor = "../data/rgb/No_Texture_Images/rt_16_5_47_cam5_notexture_basecolor.exr";


//	// Depth map
	string depth_ref_path = "../data/depth/cam1_depth.exr";
	string depth_target_path = "../data/depth/cam5_depth.exr";

	// Depth map
//	string depth_ref_path = "../data/rgb/HDdataset/rt_16_36_54_cam11_depth.exr";
//	string depth_target_path = "../data/rgb/HDdataset/rt_16_36_54_cam55_depth.exr";


	Mat depth_ref = imread(depth_ref_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	Mat depth_target = imread(depth_target_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);


//	// Metallic and Roughness
	string image_ref_MR_path = "../data/rgb/vp1_mr.png"; // store value in rgb channels,  channel b: metallic, channel green: roughness
	string image_target_MR_path = "../data/rgb/vp5_mr.png";

	// Metallic and Roughness
//	string image_ref_MR_path = "../data/rgb/HDdataset/rt_16_47_3_cam11__mr.exr"; // store value in rgb channels,  channel b: metallic, channel green: roughness
//	string image_target_MR_path = "../data/rgb/HDdataset/rt_16_47_3_cam55__mr.exr";


	Mat image_ref_baseColor= imread(image_ref_baseColor_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	Mat image_right_baseColor= imread(image_target_baseColor,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);


    // process RGB images
	Mat image_ref = imread(image_ref_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	Mat image_target = imread(image_target_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

//	// normal map GT
	string normal_GT_path="../data/rgb/normalMap/rt_23_26_45_cam1_normdir.exr";
	cv::Mat normal_map_GT = cv::imread(normal_GT_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
//	cv::Mat normal_map_GT = cv::imread("../data/rgb/normalMap/rt_20_26_27_cam11_norm.exr", cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);



	// create a metallic and roughness table for reference image
	Mat image_ref_MR= imread(image_ref_MR_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	Mat image_target_MR= imread(image_target_MR_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);


	// Camera intrinsics
	Eigen::Matrix3d K;
	Eigen::Matrix4d M;


	double fov_y= 33.398;
	double near= 0.01;
	double far= 60.0;
	double aspect= 1.333;







}

//---------------------------------------------------some notes----------------------------------------------
//	vector<Mat>ch;
//	ch.push_back(grayImage_ref);
//	ch.push_back(grayImage_ref);
//	ch.push_back(grayImage_ref);
//	Mat trible_gray_image;
//
//	merge(ch,trible_gray_image );
//	imshow("trible_gray_image",trible_gray_image);
//
////	imshow("grayImage_target",grayImage_target);
//	waitKey(0);


//	Eigen::Matrix3d R1;
//	R1<< 1,0,0,
//		0,1,0,
//		0,0,1;
//	Eigen::Vector3d t1;
//	t1<< 1,2,3;
//	Sophus::SE3d Tran(R1,t1);
//	double *transl=Tran.data();
//	cout<<"show sophus data:"<<*(transl)<<","<<*(transl+1)<<","<<*(transl+2)<<","<<*(transl+3)<<","<<*(transl+4)<<","<<*(transl+5)<<","<<*(transl+6)<<"!"<<endl;
//	Eigen::Quaternion<double> q_new(*(transl+3),*(transl),*(transl+1),*(transl+2));
//	Eigen::Matrix<double, 3,1> translation(*(transl+4),*(transl+5),*(transl+6));
//	Eigen::Matrix<double,3,3> R_new;
//	R_new=q_new.normalized().toRotationMatrix();
//	cout<<"\n show rotation matrix:"<< R_new<<endl;
//	cout<<"\n show translation"<<translation<<endl;

// read target image
//	Mat image_target = imread(image_target_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
// color space conversion
//	cvtColor(image_target, grayImage_target, COLOR_BGR2GRAY);   right
//    Eigen::Vector2i pixel(213,295);
//	imageInfo(image_target_path,pixel);



// precision improvement
//	grayImage_target.convertTo(grayImage_target, CV_64FC1, 1.0 / 255.0);

// read ref image
//	Mat image_ref = imread(image_ref_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
// color space conversion
//	cvtColor(image_ref, grayImage_ref, COLOR_BGR2GRAY);   // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11 left

// precision improvement
//	grayImage_ref.convertTo(grayImage_ref, CV_64FC1, 1.0 / 255.0);

//	imageInfo(depth_ref_path,pixel_pos);



//	depth_ref.convertTo(depth_ref_render, CV_64FC1);
//	depth_ref= depth_ref_render *(60.0-0.01) + 0.01;

//		cv::minMaxIdx(depth_ref, &min, &max);
//		cout<<"\n show the depth_ref value range:\n"<<"min:"<<min<<"max:"<<max<<endl;
//		cout<<"depth of depth_ref"<<depth_ref.depth()<<"!!!!!!!!!!!!!!"<<endl;
//   cv::minMaxIdx(depth_ref, &min, &max);
//   cout<<"\n show the depth_ref value range:\n"<<"min:"<<min<<"max:"<<max<<endl;


//   depth_target.convertTo(depth_target, CV_64F);
//   depth_target = depth_target / 5000.0;

//	depth_target.convertTo(depth_tar_render,CV_64FC1);
//	depth_target=depth_tar_render *(60.0-0.01) + 0.01;