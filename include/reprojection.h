//
// Created by cheng on 13.09.22.
//
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>
#include <visnav/local_parameterization_se3.hpp>

namespace DSONL{


	struct GetPixelGrayValue {

		GetPixelGrayValue(const double pixel_gray_val_in[1],
		                  const Eigen::Vector2d &pixelCoor,
		                  const Eigen::Matrix3d & K,
		                  const int rows,
		                  const int cols,
		                  const std::vector<double> &vec_pixel_gray_values,
		                  const std::vector<double> &img_ref_depth_values,
		                  const std::vector<double> &img_ref_vec_values
		) {
			pixel_gray_val_in_[0] = pixel_gray_val_in[0];
			rows_ = rows;
			cols_ = cols;
			pixelCoor_ = pixelCoor;
			K_ = K;
//		interp_depth = interpolated_depth;

			grid2d_depth.reset(new ceres::Grid2D<double>(&img_ref_depth_values[0],0, rows_, 0, cols_));
			interp_depth.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double>>(*grid2d_depth));


			grid2d_img_ref.reset(new ceres::Grid2D<double>(&img_ref_vec_values[0],0, rows_, 0, cols_));
			interp_img_ref.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double>>(*grid2d_img_ref));


			grid2d.reset(new ceres::Grid2D<double>(
					&vec_pixel_gray_values[0], 0, rows_, 0, cols_));
			get_pixel_gray_val.reset(
					new ceres::BiCubicInterpolator<ceres::Grid2D<double> >(*grid2d));
		}

		template<typename T>
		bool operator()(
				const T* const sT,
//			const T* const  sd, //T const *const sd,
				T *residual) const {

			Eigen::Map<Sophus::SE3<T> const> const Tran(sT);
			// project and search for optimization variable depth
			// calculate transformed pixel coordinates
			double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2);

			Eigen::Matrix<double,3,1> p_3d_no_d;
//		Eigen::Vector3d p_3d_no_d = K_.inverse() * Eigen::Matrix<double, 3, 1>(pixelCoor_(1), pixelCoor_(2), 1.0);
			p_3d_no_d<< (pixelCoor_(0)-cx)/fx, (pixelCoor_(1)-cy)/fy,1.0;
//		Eigen::Vector3d p_3d_wod, p_3d_new_proj; //((u-cx)/fx, (v-cy)/fy,1.0);
//		p_3d_wod = K_.inverse() * Eigen::Matrix<double, 3, 1>(u, v, 1.0);
			T d, u_, v_, intensity_image_ref;
			u_=(T)pixelCoor_(1);
			v_=(T)pixelCoor_(0);
			interp_depth->Evaluate(u_,v_, &d);

			interp_img_ref->Evaluate(u_,v_, &intensity_image_ref);
//		Eigen::Map<Eigen::Matrix<T, 1, 1> const> const d(sd);
//		Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(sResiduals);
			Eigen::Matrix<T, 3,1> p_w=d*p_3d_no_d;
			Eigen::Matrix<T, 3, 1> p1 = Tran * p_w ;
			Eigen::Matrix<T, 3, 1> pt = K_ * p1;

			T x = (pt[0] / pt[2]); // col id
			T y = (pt[1] / pt[2]);// row id

//	    cout<<"\n Show current col id and row id:("<<x<<","<<y<<")"<<endl;
			T pixel_gray_val_out;
			get_pixel_gray_val->Evaluate(y, x, &pixel_gray_val_out); //
//	    cout<<"\n Show current pixel_gray_val_out:"<< pixel_gray_val_out<<endl;
			residual[0] = intensity_image_ref - pixel_gray_val_out;


			return true;
		}


		double pixel_gray_val_in_[1];
		int rows_, cols_;
		Eigen::Vector2d pixelCoor_;
		Eigen::Matrix3d K_;
		std::unique_ptr<ceres::Grid2D<double> > grid2d;
		std::unique_ptr<ceres::Grid2D<double> > grid2d_depth;
		std::unique_ptr<ceres::Grid2D<double> > grid2d_img_ref;
		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double> > > get_pixel_gray_val;
		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>> >interp_depth;
		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>> >interp_img_ref;
	};

























}