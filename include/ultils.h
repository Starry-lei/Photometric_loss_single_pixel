//
// Created by cheng on 13.09.22.
//
#pragma once


#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>


#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>


#include <iostream>
#include <vector>


namespace DSONL{

	using namespace cv;
	using namespace std;

	template<typename T> Eigen::Matrix<T,3,1> backProjection(const Eigen::Matrix<T,2,1> &pixelCoord, const Eigen::Matrix<T,3,3> & K, const Mat& depthMap ){
        // convert depth map into grid2D table
		cv::Mat flat_depth_map = depthMap.reshape(1, depthMap.total() * depthMap.channels());
		std::vector<T> img_depth_values=depthMap.isContinuous() ? flat_depth_map : flat_depth_map.clone();
		std::unique_ptr<ceres::Grid2D<T> > grid2d_depth;
		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<T>> >BiCubicInterpolator_depth;

		grid2d_depth.reset(new ceres::Grid2D<double>(&img_depth_values[0],0, depthMap.rows, 0, depthMap.cols));
		BiCubicInterpolator_depth.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<T>>(*grid2d_depth));

		// get depth value at pixelCoord(col, row)
		T u, v , d;
		u= (T) pixelCoord(1);
		v= (T) pixelCoord(0);
		BiCubicInterpolator_depth->Evaluate(u,v,&d);
		T fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy = K(1, 2);
		// back projection
		Eigen::Matrix<T,3,1> p_3d_no_d;
		p_3d_no_d<< (v-cx)/fx, (u-cy)/fy,1.0;
		Eigen::Matrix<T, 3,1> p_c1=d*p_3d_no_d;
		return  p_c1;

	}


	void downscale(Mat &image, Mat &depth, Eigen::Matrix3d &K, int &level, Mat &image_d, Mat &depth_d, Eigen::Matrix3d &K_d) {
		//	imshow("depth", depth);
		//	waitKey(0);
		if (level <= 1) {
			image_d = image;
			// remove negative gray values
			image_d=cv::max(image_d,0.0);

			depth_d = depth;
			// set all nan zero
			Mat mask = Mat(depth_d != depth_d);
			depth_d.setTo(0.0, mask);

			K_d = K;
			return;
		}

		// downscale camera intrinsics

		K_d << K(0, 0) / 2.0, 0, (K(0, 2) + 0.5) / 2.0 - 0.5,
				0, K(1, 1) / 2.0, (K(1, 2) + 0.5) / 2 - 0.5,
				0, 0, 1;
		pyrDown(image, image_d, Size(image.cols / 2, image.rows / 2));
		pyrDown(depth, depth_d, Size(depth.cols / 2, depth.rows / 2));
		// remove negative gray values
		image_d=cv::max(image_d,0.0);
		// set all nan zero
		Mat mask = Mat(depth_d != depth_d);
		depth_d.setTo(0.0, mask);

		level -= 1;
		downscale(image_d, depth_d, K_d, level, image_d, depth_d, K_d);
	}

	void getNormals(const Eigen::Matrix<double,3,3> & K_, const Mat& depth){

		Mat normalsMap(depth.rows, depth.cols, CV_32FC3, Scalar(0,0,0)); // B,G,R
		double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2), f=30.0;
		// focal length: 30
		for(int x = 0; x < depth.rows; ++x)
		{
			for(int y = 0; y < depth.cols; ++y)
			{
				double d= depth.at<double>(x,y);
				Eigen::Matrix<double,3,1> p_3d_no_d;
				p_3d_no_d<< (y-cx)/fx, (x-cy)/fy,1.0;
				Eigen::Matrix<double, 3,1> p_c1=d*p_3d_no_d;

				double d_x1= depth.at<double>(x,y+1);
				double  d_y1= depth.at<double>(x+1, y);


				// calculate normal for each point
				Eigen::Matrix<double, 3,1> normal, v_x, v_y;
				v_x << (d+ (d_x1-d) *(y-cx))/f, (d_x1-d)*(x-cy)/f, (d_x1-d);
				v_y << (d_y1-d)*(y-cx)/f,(d+ (d_y1-d)*(x-cy))/f, (d_y1-d);
				normal=v_x.cross(v_y);
				normal=normal.normalized();
				Vec3f d_n((normal.z()+1)/2, normal.y(), (normal.x()+1)/2);
				Vec3f n = normalize(d_n);
				normalsMap.at<Vec3f>(x, y) = n;
			}
		}


		imshow("normals", normalsMap);
		waitKey(0);

	}


	float bilinearInterpolation(const Mat &image, const float &x, const float &y) {
		const int x1 = floor(x), x2 = ceil(x), y1 = floor(y), y2 = ceil(y);

		int width = image.cols, height = image.rows;

		//两个差值的中值
		float f12, f34;
		float epsilon = 0.0001;
		//四个临近像素坐标x像素值
		float f1, f2, f3, f4;

		if ((x < 0) || (x > width - 1) || (y < 0) || (y > height - 1)) {
			return -1.0;
		} else {
			if (fabs(x - width + 1) <= epsilon) { //如果计算点在右测边缘

				//如果差值点在图像的最右下角
				if (fabs(y - height + 1) <= epsilon) {
					f1 = image.at<float>(x1, y1);
					return f1;
				} else {
					f1 = image.at<float>(x1, y1);
					f3 = image.at<float>(x1, y2);

					//图像右方的插值
					return ((float) (f1 + (y - y1) * (f3 - f1)));
				}
			} else if (fabs(y - height + 1) <= epsilon) {
				f1 = image.at<float>(x1, y1);
				f2 = image.at<float>(x2, y1);
				return ((float) (f1 + (x - x1) * (f2 - f1)));
			} else {
				//得计算四个临近点像素值
				f1 = image.at<float>(x1, y1);
				f2 = image.at<float>(x2, y1);
				f3 = image.at<float>(x1, y2);
				f4 = image.at<float>(x2, y2);

				//第一次插值
				f12 = f1 + (x - x1) * (f2 - f1); // f(x,0)

				//第二次插值
				f34 = f3 + (x - x1) * (f4 - f3); // f(x,1)

				//最终插值
				return ((float) (f12 + (y - y1) * (f34 - f12)));
			}
		}
	}

	void printAll(const double *arr, int n) {
		cout << "show value of n:" << n << endl;
		for (int i = 0; i < n / 10; i++) {
			cout << arr[i];
			cout << ((i + 1) % 20 ? ' ' : '\n');
		}
	}

	bool removeNegativeValue(Mat& src, Mat& dst){
		dst = cv::max(src, 0);
		return true;
	}

}