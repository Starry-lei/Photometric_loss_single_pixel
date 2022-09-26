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
		GetPixelGrayValue(const double pixel_gray_val_in[1],
		                  const Eigen::Vector2d &pixelCoor,
		                  const Eigen::Matrix3d & K,
		                  const int rows,
		                  const int cols,
						  const Sophus::SE3<double> CurrentT,
		                  const std::vector<double> &vec_pixel_gray_values,
		                  const std::vector<double> &img_ref_depth_values,
		                  const std::vector<double> &img_ref_vec_values,
						  const Eigen::Vector3d & L2c1,
						  const Mat& depth_map,
						  const Mat& image_baseColor,
						  const std::vector<double> &image_metallic,
						  const std::vector<double> &image_roughness
//						  const std::vector<double> &deltaMap

		) {
			pixel_gray_val_in_[0] = pixel_gray_val_in[0];
			rows_ = rows;
			cols_ = cols;
			pixelCoor_ = pixelCoor;
			K_ = K;
			L2c1_=L2c1;
			depth_map_=depth_map;
			CurrentT_=CurrentT;


			grid2d_image_metallic.reset(new ceres::Grid2D<double>(&image_metallic[0],0, rows_, 0, cols_));
			interp_img_metallic.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double>>(*grid2d_image_metallic));

			grid2d_image_roughness.reset(new ceres::Grid2D<double>(&image_roughness[0],0, rows_, 0, cols_));
			interp_img_roughness.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double>>(*grid2d_image_roughness));



			grid2d_image_baseColor.reset(new ceres::Grid2D<uchar,3>(image_baseColor.ptr(0),0, rows_, 0, cols_));
			interp_img_baseColor.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<uchar,3>>(*grid2d_image_baseColor));


			grid2d_depth.reset(new ceres::Grid2D<double>(&img_ref_depth_values[0],0, rows_, 0, cols_));
			interp_depth.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double>>(*grid2d_depth));


			grid2d_img_ref.reset(new ceres::Grid2D<double>(&img_ref_vec_values[0],0, rows_, 0, cols_));
			interp_img_ref.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double>>(*grid2d_img_ref));


			grid2d.reset(new ceres::Grid2D<double>(&vec_pixel_gray_values[0], 0, rows_, 0, cols_));
			get_pixel_gray_val.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double> >(*grid2d));}

		template<typename T>
		bool operator()(
				const T* const sT,
//			const T* const  sd, //T const *const sd,
				T *residual) const {

			Eigen::Map<Sophus::SE3<T> const> const Tran(sT);


			// project and search for optimization variable depth
			// calculate transformed pixel coordinates
			double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2);
			// focal length: 30
			Eigen::Matrix<double,3,1> p_3d_no_d;
			p_3d_no_d<< (pixelCoor_(0)-cx)/fx, (pixelCoor_(1)-cy)/fy,1.0;

			T d, u_, v_, intensity_image_ref,d_x1,d_y1; //
			double depth_,d_x1_, d_y1_;
			u_=(T)pixelCoor_(1);
			v_=(T)pixelCoor_(0);
			interp_depth->Evaluate(u_,v_, &d);
			interp_depth->Evaluate(pixelCoor_(1),pixelCoor_(0), &depth_);

			interp_depth->Evaluate(u_,(v_+(T)1), &d_x1);
			interp_depth->Evaluate((u_+(T)1),v_, &d_y1);

			interp_depth->Evaluate(pixelCoor_(1),(pixelCoor_(0)+1.0), &d_x1_);
			interp_depth->Evaluate((pixelCoor_(1)+1.0),pixelCoor_(0), &d_y1_);

			interp_img_ref->Evaluate(u_,v_, &intensity_image_ref);
			Eigen::Matrix<T, 3,1> p_c1=d*p_3d_no_d;
			Eigen::Matrix<double,3,1> p_c1_=depth_*p_3d_no_d;

			// Eigen::Matrix<T, 3,1> p_c1= T(1.0) *backProjection<double>(pixelCoor_,K_, depth_map_);

            // calculate normal for each point
			// TODO: calculate another 5 normals for the same point
			// calculate normal for each point
			Eigen::Matrix<double, 3,1> normal, v_x, v_y;
			v_x <<  ((d_x1_-depth_)*(pixelCoor_(0)-cx)+d_x1_)/fx, (d_x1_-depth_)*(pixelCoor_(1)-cy)/fy , (d_x1_-depth_);
			v_y << (d_y1_-depth_)*(pixelCoor_(0)-cx)/fx,(d_y1_+ (d_y1_-depth_)*(pixelCoor_(1)-cy))/fy, (d_y1_-depth_);
			v_x=v_x.normalized();
			v_y=v_y.normalized();
			normal=v_x.cross(v_y);
			normal=normal.normalized();

			// Instantiation of BRDF object

			// calculate alpha_1
			Eigen::Matrix<double,3,1> alpha_1= L2c1_-p_c1_; alpha_1=alpha_1.normalized();
			// calculate beta and beta_prime;
			Eigen::Matrix<double,3,1> beta,beta_prime;
			beta=-p_c1_;
			beta=beta.normalized();
//			Eigen::Quaternion<double> q(*(sT+3),*(sT),*(sT+1),*(sT+2));
//			Eigen::Matrix<double, 3,1> translation(*(sT+4),*(sT+5),*(sT+6));
//			Eigen::Matrix<double,3,3>R;
//			R=q.normalized().toRotationMatrix();
//			R=Transf.rotationMatrix();
			beta_prime=-CurrentT_.rotationMatrix().transpose()*CurrentT_.translation()-p_c1_;
			beta_prime=beta_prime.normalized();


			double baseColor_bgr[3],metallic, roughness;
			interp_img_baseColor->Evaluate(pixelCoor_(1),pixelCoor_(0),baseColor_bgr);
			interp_img_metallic->Evaluate(pixelCoor_(1),pixelCoor_(0),&metallic);
			interp_img_roughness->Evaluate(pixelCoor_(1),pixelCoor_(0),&roughness);




			Vec3f L_(alpha_1(0),alpha_1(1),alpha_1(2));
			Vec3f N_(normal(0),normal(1),normal(2));
			Vec3f View_beta(beta(0),beta(1),beta(2));
			Vec3f View_beta_prime(beta_prime(0),beta_prime(1),beta_prime(2));
			Vec3f baseColor(baseColor_bgr[2],baseColor_bgr[1],baseColor_bgr[0]);


//			BrdfMicrofacet radiance_beta_vec(L_,N_,View_beta,(float )roughness,(float)metallic,baseColor);
//			double radiance_beta= radiance_beta_vec.radiance;
//			BrdfMicrofacet radiance_beta_prime_vec(L_,N_,View_beta_prime,(float )roughness,(float)metallic,baseColor);
//		    double radiance_beta_prime= radiance_beta_prime_vec.radiance;
//			double delta= radiance_beta/radiance_beta_prime;



			Eigen::Matrix<T, 3, 1> p1 = Tran * p_c1 ;
			Eigen::Matrix<T, 3, 1> pt = K_ * p1;

			T x = (pt[0] / pt[2]); // col id
			T y = (pt[1] / pt[2]);// row id


			T pixel_gray_val_out;
			get_pixel_gray_val->Evaluate(y, x, &pixel_gray_val_out); //

			residual[0] = intensity_image_ref - delta*pixel_gray_val_out;


			return true;
		}


		double pixel_gray_val_in_[1];
		int rows_, cols_;
		Eigen::Vector2d pixelCoor_;
		Eigen::Matrix3d K_;
		Sophus::SE3<double> CurrentT_;
		std::unique_ptr<ceres::Grid2D<double> > grid2d;
		std::unique_ptr<ceres::Grid2D<double> > grid2d_depth;
		std::unique_ptr<ceres::Grid2D<double> > grid2d_img_ref; //
		std::unique_ptr<ceres::Grid2D<uchar,3> > grid2d_image_baseColor;
		std::unique_ptr<ceres::Grid2D<double> > grid2d_image_metallic;
		std::unique_ptr<ceres::Grid2D<double> > grid2d_image_roughness;
		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double> > > get_pixel_gray_val;
		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>> >interp_depth;
		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>> >interp_img_ref;
		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<uchar ,3>> >interp_img_baseColor;
		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>> >interp_img_metallic;
		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>> >interp_img_roughness;

		Eigen::Vector3d L2c1_;
		Mat depth_map_;
	};





}