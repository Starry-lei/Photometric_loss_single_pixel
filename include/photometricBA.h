//
// Created by cheng on 13.09.22.
//
#pragma once

#include <reprojection.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>
#include <thread>

#include <ultils.h>
#include <brdfMicrofacet.h>

namespace DSONL{

	using namespace cv;
	using namespace std;

	void PhotometricBA(Mat &image, Mat &image_right, const PhotometricBAOptions &options, const Eigen::Matrix3d &K,
	                   Sophus::SE3d &pose,
					   Mat &img_ref_depth,
					   Mat & deltaMap
					   ) {
		ceres::Problem problem;
		// Setup optimization problem
		// convert rigth image into double type vector

		//cv::Mat flat_3d = normalMap.reshape(1, normalMap.total() * normalMap.channels());
		//std::vector<Vec3f> normalMap_Vec3 = normalMap.isContinuous() ? flat_3d : flat_3d.clone();
//		cv::Mat flat_metallic = metallic.reshape(1, metallic.total() * metallic.channels());
//		std::vector<double> img_metallic = metallic.isContinuous() ? flat_metallic : flat_metallic.clone();


		cv::Mat flat_deltaMap = deltaMap.reshape(1, deltaMap.total() * deltaMap.channels());
		std::vector<double> img_deltaMap = deltaMap.isContinuous() ? flat_deltaMap : flat_deltaMap.clone();

		cv::Mat flat = image_right.reshape(1, image_right.total() * image_right.channels());
		std::vector<double> img_gray_values = image_right.isContinuous() ? flat : flat.clone();

		cv::Mat flat_depth_map = img_ref_depth.reshape(1, img_ref_depth.total() * img_ref_depth.channels());
		std::vector<double> img_ref_depth_values=img_ref_depth.isContinuous() ? flat_depth_map : flat_depth_map.clone();

		cv::Mat flat_ref = image.reshape(1, image.total() * image.channels());
		std::vector<double> image_ref_vec = image.isContinuous() ? flat_ref : flat_ref.clone();


		problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters, new Sophus::test::LocalParameterizationSE3);

		std::unordered_map<int, int> inliers_filter;
		//new image
//		inliers_filter.emplace(309,294); //yes
//		inliers_filter.emplace(210,292); //yes
//		inliers_filter.emplace(209,293); //yes
//		inliers_filter.emplace(208,294); //yes
//		inliers_filter.emplace(209,295); //yes
//		inliers_filter.emplace(208,296); //yes
//		inliers_filter.emplace(206,297); //yes
		inliers_filter.emplace(173,333); //yes
		inliers_filter.emplace(378,268); //yes
//		inliers_filter.emplace(184,279); //yes
//		inliers_filter.emplace(330,354); //yes
//		inliers_filter.emplace(316,503); //yes
//		inliers_filter.emplace(312,180); //yes
		//	inliers_filter.emplace(159,294); //yes
		//	inliers_filter.emplace(256,67); //yes
		//	inliers_filter.emplace(255,69);//yes
		//	inliers_filter.emplace(252,76);//yes
		//
		//	//telephone_surface
		//	inliers_filter.emplace(254,191);//0.012 ,yes
		//	inliers_filter.emplace(243,190);//-0.0005.yes
		//	inliers_filter.emplace(241,191);//0.008 yes
		//
		//	inliers_filter.emplace(449,383);//yes
		//	inliers_filter.emplace(319,331);//yes
		//	inliers_filter.emplace(288,327);//yes
		//	inliers_filter.emplace(432,86);//yes
		//	inliers_filter.emplace(459,80);//yes
		//	inliers_filter.emplace(293,535);//yes
		//
		//  inliers_filter.emplace(310,540);//yes
		//	inliers_filter.emplace(308,555);//yes
		//	inliers_filter.emplace(307,548);//yes
		//	inliers_filter.emplace(324,376);//yes
		//	inliers_filter.emplace(324,231);//yes
		//	inliers_filter.emplace(121,93);//yes
		//	inliers_filter.emplace(131,104);//yes


		double depth_para,intensity_l ,gray_values[1]{};
		double *transformation = pose.data();
		// use pixels and depth to optimize pose and depth itself
		for (int u = 0; u< image.rows; u++) // colId, cols: 0 to 480
		{
			for (int v = 0; v < image.cols; v++) // rowId,  rows: 0 to 640
			{
				// use the inlier filter
				if(inliers_filter.count(u)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
				if(inliers_filter[u]!=v ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//				cout<<" \n show the coordinates:"<<u<<","<<v<<"---> value:"<<image.at<double>(u,v)<<endl; // checked already// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
				if (img_ref_depth.at<double>(u,v) < 1e-3 ) { continue; } //&& p_3d_new_proj(2)< 1e-4
				gray_values[0] =  image.at<double>(u, v);
				Eigen::Vector2d pixelCoord((double)v,(double)u);//  u is the row id , v is col id



				problem.AddResidualBlock(
						new ceres::AutoDiffCostFunction<GetPixelGrayValue, 1, Sophus::SE3d::num_parameters>(
								new GetPixelGrayValue(gray_values,
								                      pixelCoord,
								                      K,
								                      image.rows,
								                      image.cols,
								                      img_gray_values,
								                      img_ref_depth_values,
								                      image_ref_vec,
													  img_ref_depth,
													  img_deltaMap
								)
						),
						new ceres::HuberLoss(1), //   new ceres::HuberLoss(4.0/255.0),      // matlab (4.0/255.0)
						transformation
				);

			}
		}
		// Solve
		std::cout << "\n Solving ceres directBA ... " << endl;
		ceres::Solver::Options ceres_options;
		ceres_options.max_num_iterations = 300;
//	ceres_options.gradient_check_numeric_derivative_relative_step_size=1e-4;

		ceres_options.linear_solver_type =ceres::SPARSE_SCHUR; // ceres::SPARSE_SCHUR;  DENSE_NORMAL_CHOLESKY;
		ceres_options.num_threads = std::thread::hardware_concurrency();
		ceres_options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;

		Solve(ceres_options, &problem, &summary);
		switch (options.verbosity_level) {
			// 0: silent
			case 1:
				std::cout << summary.BriefReport() << std::endl;
				break;
			case 2:
				std::cout << summary.FullReport() << std::endl;
				break;
		}


	}

}