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

#include "ultils.h"


namespace DSONL{

	using namespace cv;
	using namespace std;

	void PhotometricBA(Mat &image, Mat &image_right, const PhotometricBAOptions &options, const Eigen::Matrix3d &K,
	                   Sophus::SE3d &pose,
					   Mat &img_ref_depth,
					   Mat deltaMap
					   ) {

		ceres::Problem problem;
		double rows_= image.rows, cols_= image.cols;

		deltaMap.convertTo(deltaMap, CV_64FC1);


		cv::Mat flat_depth_map = img_ref_depth.reshape(1, img_ref_depth.total() * img_ref_depth.channels());
		std::vector<double> img_ref_depth_values=img_ref_depth.isContinuous() ? flat_depth_map : flat_depth_map.clone();
		ceres::Grid2D<double> grid2d_depth(&img_ref_depth_values[0],0, rows_, 0, cols_);
		ceres::BiCubicInterpolator<ceres::Grid2D<double>> interpolator_depth(grid2d_depth);





		cv::Mat flat = image_right.reshape(1, image_right.total() * image_right.channels());
		std::vector<double> grayImage_right_values = image_right.isContinuous() ? flat : flat.clone();
		ceres::Grid2D<double> grid2d_grayImage_right(&grayImage_right_values[0],0, rows_, 0, cols_);


		problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters, new Sophus::test::LocalParameterizationSE3);

		std::unordered_map<int, int> inliers_filter;
		//new image
		inliers_filter.emplace(173,333); //yes
		inliers_filter.emplace(378,268); //yes


		double gray_values[1]{};
		double *transformation = pose.data();
		// use pixels,depth and delta to optimize pose and depth itself
		for (int u = 0; u< image.rows; u++) // colId, cols: 0 to 480
		{
			for (int v = 0; v < image.cols; v++) // rowId,  rows: 0 to 640
			{
				// use the inlier filter
//				if(inliers_filter.count(u)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//				if(inliers_filter[u]!=v ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~

				gray_values[0] =  image.at<double>(u, v);
				Eigen::Vector2d pixelCoord((double)v,(double)u);
				problem.AddResidualBlock(
						new ceres::AutoDiffCostFunction<GetPixelGrayValue, 1, Sophus::SE3d::num_parameters>(
								new GetPixelGrayValue(
								                      pixelCoord,
								                      K,
								                      image.rows,
								                      image.cols,
								                      grid2d_grayImage_right,
                                                      interpolator_depth,
                                                      image,
													  img_ref_depth,
													  deltaMap
								)
						),
						new ceres::HuberLoss(4/255.0),
						transformation
				);

			}
		}
		// Solve
		std::cout << "\n Solving ceres directBA ... " << endl;
		ceres::Solver::Options ceres_options;
		ceres_options.max_num_iterations = 300;

		ceres_options.linear_solver_type =ceres::SPARSE_SCHUR;
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