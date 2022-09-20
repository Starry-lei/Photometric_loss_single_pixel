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
#include <ultils.h>
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
		GetPixelGrayValue(const double pixel_gray_val_in[1],
		                  const Eigen::Vector2d &pixelCoor,
		                  const Eigen::Matrix3d & K,
		                  const int rows,
		                  const int cols,
		                  const std::vector<double> &vec_pixel_gray_values,
		                  const std::vector<double> &img_ref_depth_values,
		                  const std::vector<double> &img_ref_vec_values,
						  const Eigen::Vector3d & light_source,
						  const Mat& depth_map
		) {
			pixel_gray_val_in_[0] = pixel_gray_val_in[0];
			rows_ = rows;
			cols_ = cols;
			pixelCoor_ = pixelCoor;
			K_ = K;
			light_source_=light_source;
			depth_map_=depth_map;

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
			double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2), f=30.0;
			// focal length: 30
			Eigen::Matrix<double,3,1> p_3d_no_d;
			p_3d_no_d<< (pixelCoor_(0)-cx)/fx, (pixelCoor_(1)-cy)/fy,1.0;

			T d, d_x1,d_y1, u_, v_, intensity_image_ref;
			u_=(T)pixelCoor_(1);
			v_=(T)pixelCoor_(0);
			interp_depth->Evaluate(u_,v_, &d);
			interp_depth->Evaluate(u_,(v_+(T)1), &d_x1);
			interp_depth->Evaluate((u_+(T)1),v_, &d_y1);

			interp_img_ref->Evaluate(u_,v_, &intensity_image_ref);
			Eigen::Matrix<T, 3,1> p_c1=d*p_3d_no_d;
//			Eigen::Matrix<T, 3,1> p_c1= T(1.0) *backProjection<double>(pixelCoor_,K_, depth_map_);

            // look up normal for each point







			// calculate alpha_1
			Eigen::Matrix<T, 3,1> alpha_1;
			alpha_1= light_source_-p_c1;
			// calculate beta and beta_prime;
			Eigen::Matrix<T,3,1> beta,beta_prime;
			beta=-p_c1;
			Eigen::Quaternion<T> q(*(sT+3),*(sT),*(sT+1),*(sT+2));
			Eigen::Matrix<T, 3,1> translation(*(sT+4),*(sT+5),*(sT+6));
			Eigen::Matrix<T,3,3>R;
			R=q.normalized().toRotationMatrix();
			beta_prime=-R.transpose()*translation-p_c1;








//			beta_prime=;


			Eigen::Matrix<T, 3, 1> p1 = Tran * p_c1 ;
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
		Eigen::Vector3d light_source_;
		Mat depth_map_;
	};





}