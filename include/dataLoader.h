//
// Created by cheng on 10.10.22.
//

#pragma once

#include "setting.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>

#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <iostream>

#define baseline_l 0;
#define baseline_s 1;

namespace DSONL{
	using namespace cv;
	using namespace std;

	Eigen::Matrix<double,3,1> light_source(7.1, -22.9 ,11.0);
	Eigen::Matrix<double,3,1> Camera1(3.8, -16.5 ,26.1);
	Eigen::Matrix<double,3,1> Camera2(0.3 ,-16.9, 27.7);

	Eigen::Vector3d l_w(0.223529, 0.490196, 0.843137);
	Eigen::Vector3d N_w(0.0352942, -0.223529, 0.976471);


	struct dataOptions {
		/// 0: big baseline
		/// 1: small baseline
		/// 2: smaller baseline(no specular light on the wall, i.e, lower metallic on wall)
		/// 3: MicroBaseline
		/// 4:  Control Experiment for Lambertian Data  Non-Lambertian Data

		int baseline = 4;
		/// is textured or not
		bool isTextured = true;
		/// use gree channel for testing
		int channelIdx= 1;

		bool lambertian= true;

		bool remove_outlier_manually= true;
		/// should we calculate 3 channel delta map for loss function now?????????????????

	};

	class dataLoader{


	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		dataLoader();
		~dataLoader();
		dataOptions options_;
		Mat image_ref_metallic;
		Mat image_ref_roughness;
		Mat grayImage_ref;
		Mat grayImage_target;
		Mat depth_map_ref;
		Mat depth_map_target;
		Mat outlier_mask_big_baseline;
		Eigen::Matrix3f camera_intrinsics;
		Eigen::Matrix4d M_matrix;
		Mat normal_map_GT;
		Mat image_ref_baseColor;
		Mat image_target_baseColor;
		Eigen::Matrix3d R12;
		Eigen::Vector3d t12;
		Eigen::Quaterniond q_12;
		Eigen::Matrix3d R1;
		Eigen::Matrix3d R2;
		int rows;
		int cols;

		void Init(){

			// Camera intrinsics
			camera_intrinsics<< 800.0, 0, 320,
								0, 800.0, 240,
								0,   0,  1;

			if(options_.isTextured){

				// RGB image with texture
//				string image_ref_path =            "../data/rgb/Texture_Image/rt_17_3_40_cam1_texture.exr";//rt_13_53_33_cam1_rgb
//				string image_ref_path = "../data/rgb/Texture_Image/rt_13_53_33_cam1_rgb.exr";//rt_16_18_41_cam1_rgb


				// BaseColor Image with texture
//				string image_ref_baseColor_path = "../data/rgb/Texture_Image/rt_17_4_52_cam1_texture_basecolor.exr";//rt_13_56_28_cam1_basecolor


				//	// Metallic and Roughness------------------------not replaced!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
//				string image_ref_MR_path = "../data/rgb/vp1_mr.png"; // store value in rgb channels,  channel b: metallic, channel green: roughness

				// Depth map
//				string depth_ref_path = "../data/depth/cam1_depth.exr";
				string image_ref_path;
				string lambertian_image_ref_path;
				string image_ref_baseColor_path;
				string image_ref_MR_path;
				string lambertian_image_ref_MR_path;
				string depth_ref_path;


				if (options_.baseline==2){
					 image_ref_path = "../data/rgb/Texture_Image/rt_16_18_41_cam1_rgb.exr";
					 image_ref_baseColor_path = "../data/rgb/Texture_Image/rt_13_56_28_cam1_basecolor.exr";
					 image_ref_MR_path = "../data/rgb/cam1_mr16h29m.png";
					 depth_ref_path = "../data/depth/rt_13_54_33_cam1_depth.exr";

				}else if(options_.baseline==3){
					image_ref_path = "../data/rgb/Texture_Image/rt_7_55_54_cam1_rgb.exr";
					image_ref_baseColor_path = "../data/rgb/Texture_Image/rt_7_56_53_cam1_basecolor.exr";
					image_ref_MR_path="../data/rgb/Texture_Image/mr_data/cam1_mr07h59m.png";
					depth_ref_path = "../data/depth/rt_8_0_9_cam1_depth.exr";

				} else if (options_.baseline==4){
					image_ref_path ="../data/rgb/ControlExperiment/leftCamera/nonlambertian/specular_rt_18_41_46_cam1_rgb.exr";
					lambertian_image_ref_path= "../data/rgb/ControlExperiment/leftCamera/rt_18_40_2_cam1_rgb.exr";

					image_ref_baseColor_path="../data/rgb/ControlExperiment/leftCamera/rt_18_41_19_cam1_basecolor.exr";
					depth_ref_path="../data/rgb/ControlExperiment/leftCamera/rt_17_51_15_cam1_depth.exr";

					image_ref_MR_path="../data/rgb/ControlExperiment/leftCamera/nonlambertian/cam1_mr18h46m.png";
					lambertian_image_ref_MR_path="../data/rgb/ControlExperiment/leftCamera/cam1_mr18h43m.png";

				}




				//normal map GT
				string normal_GT_path="../data/rgb/normalMap/rt_23_26_45_cam1_normdir.exr";


				Mat image_ref = imread(image_ref_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
				Mat lambertian_image_ref = imread(lambertian_image_ref_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

				Mat depth_ref = imread(depth_ref_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
				image_ref_baseColor= imread(image_ref_baseColor_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
				// create a metallic and roughness table for reference image
				Mat image_ref_MR= imread( image_ref_MR_path ,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
				Mat lambertian__image_ref_MR= imread(lambertian_image_ref_MR_path ,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
				normal_map_GT = imread(normal_GT_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);



				Mat ref_mr_table[3];

//				imshow("image_ref",image_ref);
//				imshow("lambertian_image_ref",lambertian_image_ref);
//				waitKey(0);

				if (options_.lambertian){
					image_ref=lambertian_image_ref.clone();
					image_ref_MR=lambertian__image_ref_MR.clone();
				}

				split(image_ref_MR,ref_mr_table);// 0: red, 1: green, 2: blue
				Mat image_ref_metallic_=  ref_mr_table[2];
				image_ref_metallic_.convertTo(image_ref_metallic, CV_32FC1,1.0 / 255.0);

				Mat image_ref_roughness_= ref_mr_table[1];
				image_ref_roughness_.convertTo(image_ref_roughness, CV_32FC1,1.0 / 255.0);

				int channelIdx= options_.channelIdx;

				extractChannel(image_ref, grayImage_ref, channelIdx);

				rows=image_ref.rows;
				cols=image_ref.cols;
				setGlobalCalib(cols,rows,camera_intrinsics);


				// left map depth
				Mat channel[3],depth_ref_render, channel_tar[3], depth_tar_render;
				split(depth_ref,channel);
				depth_ref=channel[0];
				depth_ref.convertTo(depth_map_ref, CV_64FC1);

				string image_target_path;
				string lambertian_image_target_path;
				string image_target_baseColor_path;
				string depth_target_path;
				string image_target_MR_path;
				string lambertian_image_target_MR_path;
				string outlier_mask_big_baseline_path;
				Eigen::Quaterniond q_1(0.009445649,-0.0003128,-0.9994076,-0.0330920); //  cam1  wxyz
				Eigen::Vector3d t1( 3.8, -16.5, 26.1);
				R1=q_1.toRotationMatrix();

				if(options_.baseline==0){
					 image_target_path = "../data/rgb/Texture_Image/rt_17_3_40_cam5_texture.exr";
					image_target_baseColor_path = "../data/rgb/Texture_Image/rt_17_4_52_cam5_texture_basecolor.exr";
					 depth_target_path = "../data/depth/cam5_depth.exr";
					 image_target_MR_path = "../data/rgb/vp5_mr.png";


					Eigen::Quaterniond q_2(-0.08078633,-0.0084485,-0.9962677,-0.0292077 ); //  cam5  wxyz
					Eigen::Vector3d t2(-5.1,-15.2 ,27.5);
					R2=q_2.toRotationMatrix();
					R12= R2.transpose() * R1;
					t12= R2.transpose()* (t1-t2);
					q_12= R12;



				}else if (options_.baseline==1){
//					 image_target_path = "../data/rgb/small_baseline/rt_11_3_21_cam6_rgb.exr";//rt_13_53_33_cam6_rgb
//					image_target_path = "../data/rgb/small_baseline/rt_13_53_33_cam6_rgb.exr"; //rt_16_18_41_cam6_rgb
					image_target_path = "../data/rgb/small_baseline/rt_16_18_41_cam6_rgb.exr"; //


//					image_target_baseColor ="../data/rgb/small_baseline/rt_12_56_23_cam6_basecolor.exr";//rt_13_56_28_cam6_basecolor
					image_target_baseColor_path ="../data/rgb/small_baseline/rt_13_56_28_cam6_basecolor.exr";//

//					depth_target_path="../data/rgb/small_baseline/rt_12_44_34_cam6_depth.exr";//rt_13_54_33_cam6_depth
					depth_target_path="../data/rgb/small_baseline/rt_13_54_33_cam6_depth.exr";//

//					image_target_MR_path = "../data/rgb/small_baseline/cam6_mr.png";//cam6_mr16h30m
					image_target_MR_path = "../data/rgb/small_baseline/cam6_mr16h30m.png";//


					outlier_mask_big_baseline_path="../data/rgb/Texture_Image/red_mask.png"; // 0 means outlier
					outlier_mask_big_baseline= imread(outlier_mask_big_baseline_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);


					Eigen::Quaterniond q_2(-0.0146847,-0.0064972,-0.9994298,-0.0297027 ); //  cam6  wxyz
					 Eigen::Vector3d t2(0,-16.5,28.2);
					 R2=q_2.toRotationMatrix();

					 R12= R2.transpose() * R1;
					 t12= R2.transpose()* (t1-t2);
					 q_12= R12;



				}else if (options_.baseline==2){

//					image_target_path = "../data/rgb/cam7_smaller_baseline/rt_16_33_27_cam7_rgb.exr";//rt_13_53_33_cam7_rgb
//					image_target_path = "../data/rgb/cam7_smaller_baseline/rt_13_53_33_cam7_rgb.exr";//rt_16_18_41_cam7_rgb
					image_target_path = "../data/rgb/cam7_smaller_baseline/rt_16_18_41_cam7_rgb.exr";//

//					image_target_baseColor ="../data/rgb/cam7_smaller_baseline/rt_23_4_36_cam7_basecolor.exr";//rt_13_56_28_cam7_basecolor
					image_target_baseColor_path ="../data/rgb/cam7_smaller_baseline/rt_13_56_28_cam7_basecolor.exr";//


//					depth_target_path="../data/rgb/cam7_smaller_baseline/rt_23_2_30_cam7_depth.exr";//rt_13_54_33_cam7_depth
					depth_target_path="../data/rgb/cam7_smaller_baseline/rt_13_54_33_cam7_depth.exr";//

//					image_target_MR_path = "../data/rgb/cam7_smaller_baseline/cam7_mr.png";//cam7_mr16h30m
					image_target_MR_path = "../data/rgb/cam7_smaller_baseline/cam7_mr16h30m.png";//

					Eigen::Quaterniond q_2(-0.0146847,-0.0064972,-0.9994298,-0.0297027 ); //  cam7  wxyz
					Eigen::Vector3d t2(4.3,-17,26.6);
					R2=q_2.toRotationMatrix();

					R12= R2.transpose() * R1;
					t12= R2.transpose()* (t1-t2);
					q_12= R12;

				}else if (options_.baseline==3){
					/// micro baseline
					image_target_path = "../data/rgb/microBaseline/rt_7_55_54_cam5_rgb.exr";
					image_target_baseColor_path = "../data/rgb/microBaseline/rt_7_56_53_cam5_basecolor.exr";
					depth_target_path="../data/rgb/microBaseline/rt_5_36_57_cam5_depth.exr";
					image_target_MR_path="../data/rgb/microBaseline/cam5_mr07h58m.exr";
					Eigen::Quaterniond q_2(-0.004485924,-0.0061938,-0.9995276,-0.0297675 ); //  cam7  wxyz
					Eigen::Vector3d t2(3.3,-16.6,26);
					R2=q_2.toRotationMatrix();
					R12= R2.transpose() * R1;
					t12= R2.transpose()* (t1-t2);
					q_12= R12;
				}else if (options_.baseline==4){
					image_target_path ="../data/rgb/ControlExperiment/rightCamera/nonlambertian/specular_rt_18_41_46_cam4_rgb.exr";
					lambertian_image_target_path="../data/rgb/ControlExperiment/rightCamera/rt_18_40_2_cam4_rgb.exr";

					image_target_baseColor_path = "../data/rgb/ControlExperiment/rightCamera/rt_18_41_19_cam4_basecolor.exr";
					depth_target_path = "../data/rgb/ControlExperiment/rightCamera/rt_17_51_15_cam4_depth.exr";

					image_target_MR_path = "../data/rgb/ControlExperiment/rightCamera/nonlambertian/cam4_mr18h45m.png";
					lambertian_image_target_MR_path="../data/rgb/ControlExperiment/rightCamera/cam4_mr18h44m.png";


					//w:0.02374156;x:-0.0053507;y:-0.9992557;z:-0.0299305 X:4 Y:-16.4 Z:26

					Eigen::Quaterniond q_2(0.02374156,-0.0053507,-0.9992557,-0.0299305 ); //  wxyz
					Eigen::Vector3d t2(4,-16.4,26);
					R2=q_2.toRotationMatrix();
					R12= R2.transpose() * R1;
					t12= R2.transpose()* (t1-t2);
					q_12= R12;




				}




				Mat image_target = imread(image_target_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
				Mat lambertian_image_target=imread(lambertian_image_target_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

				Mat depth_target = imread(depth_target_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
				image_target_baseColor= imread(image_target_baseColor_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

				Mat image_target_MR= imread(image_target_MR_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
				Mat lambertian_image_target_MR=imread(lambertian_image_target_MR_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);


				Mat taget_mr_table[3];
				split(image_target_MR,taget_mr_table);// 0: red, 1: green, 2: blue
				Mat image_target_metallic=  taget_mr_table[2];
				Mat image_target_roughness= taget_mr_table[1];
				image_target_metallic.convertTo(image_target_metallic, CV_32FC1,1.0 / 255.0);
				image_target_roughness.convertTo(image_target_roughness, CV_32FC1,1.0 / 255.0);



				if (options_.lambertian){
					image_target=lambertian_image_target.clone();// !!!!!!!!!!!!!!!!!!!!
					image_target_MR=lambertian_image_target_MR.clone();// !!!!!!!!!!!!!!!!
				}
				extractChannel(image_target, grayImage_target, channelIdx);
				// right map depth
				split(depth_target, channel_tar);
				depth_target=channel_tar[0];
				depth_target.convertTo(depth_map_target, CV_64FC1);

			}else{
				// RGB image without texture
					string image_ref_path = "../data/rgb/No_Texture_Images/rt_16_4_56_cam1_notexture.exr";
				    string image_ref_baseColor_path = "../data/rgb/No_Texture_Images/rt_16_5_47_cam1_notexture_basecolor.exr";
				if(options_.baseline==0){
					string image_target_path = "../data/rgb/No_Texture_Images/rt_16_4_56_cam5_notexture.exr";
					string image_target_baseColor = "../data/rgb/No_Texture_Images/rt_16_5_47_cam5_notexture_basecolor.exr";
				}else if (options_.baseline==1){
					// TODO: small baseline data
					string image_target_path = "../data/rgb/No_Texture_Images/rt_16_4_56_cam5_notexture.exr";
				}


			}


		}




	};

	dataLoader::dataLoader(void ) {
		cout<<"loading data..."<<endl;
	}

	dataLoader::~dataLoader(void ) {
		cout<<"The program ends here, have a nice day!"<<endl;
	}

 void showImage(Mat& image, string image_name){
	imshow(image_name, image);

 }












}

//---------------------------------------------------some notes----------------------------------------------

//	// HD RGB image with texture
//	string image_ref_path = "../data/rgb/HDdataset/rt_16_40_56_cam11__rgb.exr";
//	string image_target_path = "../data/rgb/HDdataset/rt_16_40_56_cam55__rgb.exr";

// HD BaseColor Image with texture
//	string image_ref_baseColor_path = "../data/rgb/HDdataset/rt_16_35_53_cam11__basecolor.exr";
//	string image_target_baseColor = "../data/rgb/HDdataset/rt_16_35_53_cam55__basecolor.exr";

// HD Depth map
//	string depth_ref_path = "../data/rgb/HDdataset/rt_16_36_54_cam11_depth.exr";
//	string depth_target_path = "../data/rgb/HDdataset/rt_16_36_54_cam55_depth.exr";
// Metallic and Roughness
//	string image_ref_MR_path = "../data/rgb/HDdataset/rt_16_47_3_cam11__mr.exr"; // store value in rgb channels,  channel b: metallic, channel green: roughness
//	string image_target_MR_path = "../data/rgb/HDdataset/rt_16_47_3_cam55__mr.exr";




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

//			double fov_y= 33.398;
//			double near= 0.01;
//			double far= 60.0;
//			double aspect= 1.333;
//
//			M_matrix << 1.0/(tan(0.5*fov_y)*aspect), 0, 0, 0,
//					0,  atan(0.5*fov_y), 0   ,  0,
//					0,0, (far+near)/(near-far), 2*far*near/(near-far),
//					0,  0,   -1,    0;