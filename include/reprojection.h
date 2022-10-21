//
// Created by cheng on 13.09.22.
//
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>
#include <visnav/local_parameterization_se3.hpp>
#include "ultils.h"
#include "brdfMicrofacet.h"

#include <opencv2/core/eigen.hpp>


namespace DSONL{

	struct PhotometricBAOptions {
		/// 0: silent, 1: ceres brief report (one line), 2: ceres full report
		int verbosity_level = 1;

		/// update intrinsics or keep fixed
		bool optimize_intrinsics = false;

		/// use huber robust norm or squared norm
		bool use_huber = true;

		/// parameter for huber loss (in pixel)
		float huber_parameter = 1.0;

		/// maximum number of solver iterations
		int max_num_iterations = 20;
	};

	struct GetPixelGrayValue {

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		GetPixelGrayValue(
		                  const Eigen::Vector2d &pixelCoor,
		                  const Eigen::Matrix3d & K,
		                  const int rows,
		                  const int cols,
		                  ceres::Grid2D<double>& grid2d_grayImage_right ,
                          ceres::BiCubicInterpolator<ceres::Grid2D<double>>& interpolator_depth,
		                  const Mat& gray_Image_ref,
						  const Mat& depth_map,
						  const Mat& deltaMap

		) {
			rows_ = rows;
			cols_ = cols;
			pixelCoor_ = pixelCoor;
			K_ = K;

			interpolator_depth.Evaluate(pixelCoor_(1),pixelCoor_(0), &depth_val);
			delta_val=deltaMap.at<double>(pixelCoor_(1),pixelCoor_(0));
			gray_Image_ref_val=gray_Image_ref.at<double>(pixelCoor_(1),pixelCoor_(0));

			get_pixel_gray_val.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double> >(grid2d_grayImage_right));


		}

		template<typename T>
		bool operator()(
				const T* const sT,
//			const T* const  sd, //T const *const sd,
				T *residual) const {

			Eigen::Map<Sophus::SE3<T> const> const Tran(sT);
			double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2);
			Eigen::Matrix<double,3,1> p_3d_no_d;
			p_3d_no_d<< (pixelCoor_(0)-cx)/fx, (pixelCoor_(1)-cy)/fy,1.0;
			T d, u_, v_, intensity_image_ref,d_x1,d_y1, delta;
			u_=(T)pixelCoor_(1);
			v_=(T)pixelCoor_(0);

			d= (T) depth_val;

			intensity_image_ref=(T) gray_Image_ref_val;
			Eigen::Matrix<T, 3,1> p_c1=d*p_3d_no_d;

			double delta_falg;
			delta_falg=delta_val;
			delta=(T)delta_val;


			Eigen::Matrix<T, 3, 1> p1 = Tran * p_c1 ;
			Eigen::Matrix<T, 3, 1> pt = K_ * p1;

			T x = (pt[0] / pt[2]); // col id !
			T y = (pt[1] / pt[2]);// row id !

			if (x> (T)0 && x< (T)cols_ && y>(T)0 && y<(T)rows_ ){
//				if(delta_falg>1.2 || delta_falg < 0.9){cout<<"now, in the ceres loss function we show delta value:"<<delta_falg<<endl;}
				T pixel_gray_val_out;
				get_pixel_gray_val->Evaluate(y, x, &pixel_gray_val_out);
				residual[0] = delta*intensity_image_ref - pixel_gray_val_out;
				return true;
			}


		}


		int rows_, cols_;
		Eigen::Vector2d pixelCoor_;
		Eigen::Matrix3d K_;
		Sophus::SE3<double> CurrentT_;

		double depth_val;
		double delta_val;
		double gray_Image_ref_val;

		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> get_pixel_gray_val;



	};





}