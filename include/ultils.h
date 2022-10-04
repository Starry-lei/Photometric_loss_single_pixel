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
#include <cmath>
#include <omp.h>

#include <pcl/io/ply_io.h>

#include <pcl/io/pcd_io.h>
#include <chrono>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace DSONL{

	using namespace cv;
	using namespace std;
	const double DEG_TO_ARC = 0.0174532925199433;





    Eigen::Matrix3d rotmatz(double a)
    {
	    Eigen::Matrix3d R;
		R<<cos(a), -sin(a), 0,
		   sin(a) , cos(a), 0,
	         0,     0,      1;
	    return R;
	}
	Eigen::Matrix3d rotmatx(double a)
	{
		Eigen::Matrix3d R;
		R<<     1,     0,   0,
				0,cos(a), -sin(a),
				0,   sin(a) , cos(a);
		return R;
	}

	Eigen::Matrix3d rotmaty(double a)
	{
		Eigen::Matrix3d R;
		R<<      cos(a) ,0,sin(a),
		          0,    1 ,    0,
				-sin(a), 0, cos(a);
		return R;
	}
	Eigen::Vector3d light_C1( Eigen::Vector3d light_w){




//
//		Eigen::Matrix3d R_X=  rotmatx(-3.793*DEG_TO_ARC);
//		Eigen::Matrix3d R_Y=  rotmaty(-178.917*DEG_TO_ARC);
//		Eigen::Matrix3d R_Z=  rotmatz(0*DEG_TO_ARC);

		Eigen::Quaterniond q(0.009445649,-0.0003128,-0.9994076,-0.0330920);





//	    Eigen::Matrix3d R_1w=  R_Y*R_X*R_Z;
//
//		Eigen::Matrix3d R_1w_new = (Eigen::AngleAxisd(-178.917*DEG_TO_ARC, Eigen::Vector3d::UnitY()) *
//		                   Eigen::AngleAxisd(0*DEG_TO_ARC, Eigen::Vector3d::UnitZ()) *
//		                   Eigen::AngleAxisd(-3.793*DEG_TO_ARC, Eigen::Vector3d::UnitX())).toRotationMatrix();




		Eigen::Vector3d t_1w;
		t_1w<<3.8, -16.5, 26.1;

//		Eigen::Matrix3d  R_w1;
//		Eigen::Vector3d  t_w1;
//		R_w1=R_1w.transpose();
//        R_w1=(R_Y*R_X*R_Z).transpose();

//		t_w1= - R_1w.transpose()*t_1w;
//		t_w1=- (R_Y*R_X*R_Z).transpose()*t_1w;


//		return (R_w1* light_w+ t_w1);

		return (q.toRotationMatrix()).transpose()* (light_w -t_1w);

	}






   class normalMapFiltering {
   private:
	   // member function to pad the image before convolution
	   Mat padding(Mat img, int k_width, int k_height, string type) {
		   Mat scr;
		   img.convertTo(scr, CV_64FC1);
		   int pad_rows, pad_cols;
		   pad_rows = (k_height - 1) / 2;
		   pad_cols = (k_width - 1) / 2;
		   Mat pad_image(Size(scr.cols + 2 * pad_cols, scr.rows + 2 * pad_rows), CV_64FC1, Scalar(0));
		   scr.copyTo(pad_image(Rect(pad_cols, pad_rows, scr.cols, scr.rows)));
		   // mirror padding
		   if (type == "mirror") {
			   for (int i = 0; i < pad_rows; i++) {
				   scr(Rect(0, pad_rows - i, scr.cols, 1)).copyTo(pad_image(Rect(pad_cols, i, scr.cols, 1)));
				   scr(Rect(0, (scr.rows - 1) - pad_rows + i, scr.cols, 1)).copyTo(pad_image(Rect(pad_cols,
				                                                                                  (pad_image.rows - 1) -
				                                                                                  i, scr.cols, 1)));
			   }

			   for (int j = 0; j < pad_cols; j++) {
				   pad_image(Rect(2 * pad_cols - j, 0, 1, pad_image.rows)).copyTo(
						   pad_image(Rect(j, 0, 1, pad_image.rows)));
				   pad_image(Rect((pad_image.cols - 1) - 2 * pad_cols + j, 0, 1, pad_image.rows)).
						   copyTo(pad_image(Rect((pad_image.cols - 1) - j, 0, 1, pad_image.rows)));
			   }

			   return pad_image;
		   }
			   // replicate padding
		   else if (type == "replicate") {
			   for (int i = 0; i < pad_rows; i++) {
				   scr(Rect(0, 0, scr.cols, 1)).copyTo(pad_image(Rect(pad_cols, i, scr.cols, 1)));
				   scr(Rect(0, (scr.rows - 1), scr.cols, 1)).copyTo(pad_image(Rect(pad_cols,
				                                                                   (pad_image.rows - 1) - i, scr.cols,
				                                                                   1)));
			   }

			   for (int j = 0; j < pad_cols; j++) {
				   pad_image(Rect(pad_cols, 0, 1, pad_image.rows)).copyTo(pad_image(Rect(j, 0, 1, pad_image.rows)));
				   pad_image(Rect((pad_image.cols - 1) - pad_cols, 0, 1, pad_image.rows)).
						   copyTo(pad_image(Rect((pad_image.cols - 1) - j, 0, 1, pad_image.rows)));
			   }
			   // zero padding
			   return pad_image;
		   } else {
			   return pad_image;
		   }

	   }


	   // member function to define kernels for convolution
	   Mat define_kernel(int k_width, int k_height, string type)
	   {
		   // box kernel
		   if (type == "box")
		   {
			   Mat kernel(k_height, k_width, CV_64FC1, Scalar(1.0 / (k_width * k_height)));
			   return kernel;
		   }
			   // gaussian kernel
		   else if (type == "gaussian")
		   {
			   // I will assume k = 1 and sigma = 1
			   int pad_rows = (k_height - 1) / 2;
			   int pad_cols = (k_width - 1) / 2;
			   Mat kernel(k_height, k_width, CV_64FC1);
			   for (int i = -pad_rows; i <= pad_rows; i++)
			   {
				   for (int j = -pad_cols; j <= pad_cols; j++)
				   {
					   kernel.at<double>(i + pad_rows, j + pad_cols) = exp(-(i*i + j*j) / 2.0);
				   }
			   }

			   kernel = kernel /sum(kernel).val[0];
			   return kernel;
		   }
	   }

   public:

	   void convolve(Mat scr, Mat &dst, int k_w, int k_h, string paddingType, string filterType)
	   {
		   Mat pad_img, kernel;
		   pad_img = padding(scr, k_w, k_h, paddingType);
		   kernel = define_kernel(k_w, k_h, filterType);

		   Mat output = Mat::zeros(scr.size(), CV_64FC1);

		   for (int i = 0; i < scr.rows; i++)
		   {
			   for (int j = 0; j < scr.cols; j++)
			   {
				   output.at<double>(i, j) = sum(kernel.mul(pad_img(Rect(j, i, k_w, k_h)))).val[0];
			   }
		   }

//		   output.convertTo(dst, CV_8UC1);
			dst=output;
	   }

   };





	// read GT function

	void readGT(string& img_path,  Eigen::Vector2i& position ){

//		position,  fst is rowIdx, snd is colIdx
		Mat im= imread(img_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
       int img_depth= im.depth();
		switch (img_depth) {
			case 5:
				cout<<"The data type of current image is CV_32F. \n"<<endl;

		}

		cout<<"\n show Value depth:\n"<<im.depth()<<"\n show Val channels :\n "<< im.channels()<<endl;
		imshow("getValues", im);

		double min_v, max_v;
		cv::minMaxLoc(im, &min_v, &max_v);
		cout<<"\n show Value min, max:\n"<<min_v<<","<<max_v<<endl;

//		double fstChannel, sndChannel,thrChannel;

		if (static_cast<int>(im.channels())==1)
		{
			//若为灰度图，显示鼠标点击的坐标以及灰度值
			cout<<"at("<<position.x()<<","<<position.y()<<")value is:"<<static_cast<float >(im.at<float>(position.x(),position.y()))<<endl;
		}
		else if (static_cast<int>(im.channels() == 3))
		{
			//若图像为彩色图像，则显示鼠标点击坐标以及对应的B, G, R值
			cout << "at (" << position.x() << ", " << position.y() << ")"
			     << "  R value is: " << static_cast<float>(im.at<Vec3f>(position.x(), position.y())[2])
				 << "  G value is: " << static_cast<float>(im.at<Vec3f>(position.x(), position.y())[1])
				 << "  B value is: " << static_cast<float >(im.at<Vec3f>(position.x(), position.y())[0])
			     << endl;
		}

		waitKey(0);

	}















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


	Mat singleNormalFiltering(Mat& normalMap, int k_w, int k_h ){

		Mat newNormalsMap(normalMap.rows, normalMap.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R

		vector<Mat> channels_new, channels_result;
		Mat normal_z, normal_y,normal_x, normal_z_new, normal_y_new, normal_x_new;
		Mat channels[3];
		split(normalMap,channels);
		normal_z= channels[0];
		normal_y= channels[1];
		normal_x= channels[2];


		normalMapFiltering normalFilter;

		normalFilter.convolve(normal_z,normal_z_new, k_w, k_h, "zero", "box"); // zero ,mirror ,  replicate ;   gaussian, box
		normalFilter.convolve(normal_y,normal_y_new, k_w, k_h, "zero", "box");
		normalFilter.convolve(normal_x,normal_x_new, k_w, k_h, "zero", "box");

		Mat norm_mat;
		sqrt((normal_z_new.mul(normal_z_new)+ normal_y_new.mul(normal_y_new) + normal_x_new.mul(normal_x_new)), norm_mat);

		channels_result.push_back(normal_z_new/norm_mat);
		channels_result.push_back(normal_y_new/norm_mat);
		channels_result.push_back(normal_x_new/norm_mat);
		merge(channels_result,newNormalsMap);

		return  newNormalsMap;

	}

	Mat normalMapFilter(Mat& normalMap){

		Mat newNormalsMap(normalMap.rows, normalMap.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R
		newNormalsMap= singleNormalFiltering(normalMap, 5,5);
//		Mat normalFilter2 = singleNormalFiltering(newNormalsMap, 7, 7);
//		imshow("normals", normalMap);
//		imshow("first filter 5*5", newNormalsMap);
//		imshow(" second filter 7*7", normalFilter2);
//		waitKey(0);

		return newNormalsMap;

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

	Eigen::Vector3d backprojection_renderDepth(double& depth, int& p_x, int& p_y, Eigen::Matrix<double,4,4>& M){



	}

	Eigen::Vector3d backprojection_realDepth(double& depth, int& p_row, int& p_col, Eigen::Matrix<double,3,3>& K_){

		double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2), f=30.0;
		Eigen::Matrix<double,3,1> p_3d_no_d;
		p_3d_no_d<< (p_col-cx)/fx, (p_row-cy)/fy,1.0;
		Eigen::Matrix<double, 3,1> p_c1=depth*p_3d_no_d;

		return  p_c1;

	}

	Mat getNormals(const Eigen::Matrix<double,3,3> & K_, const Mat& depth){

		Mat normalsMap(depth.rows, depth.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R
		Mat normalsMap_bgr(depth.rows, depth.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R
		double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2), f=30.0;
		// focal length: 30
		cout <<"fx:"<<fx <<"fy:"<<fy<<endl;
		std::unordered_map<int, int> inliers_filter;


//		inliers_filter.emplace(229, 335); //yes
//		inliers_filter.emplace(232, 333); //yes
//		inliers_filter.emplace(234, 335); //yes


		for(int x = 0; x < depth.rows; ++x)
		{
			for(int y = 0; y < depth.cols; ++y)
			{
//				if(inliers_filter.count(x)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//		        if(inliers_filter[x]!=y ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//				cout<<" \n show the coordinates:"<<x<<","<<y<<"---> value:"<<depth.at<double>(x,y)<<endl;
				double d= depth.at<double>(x,y);
				Eigen::Matrix<double,3,1> p_3d_no_d;
				p_3d_no_d<< (y-cx)/fx, (x-cy)/fy,1.0;
				Eigen::Matrix<double, 3,1> p_c1=d*p_3d_no_d;
				double d_x1= depth.at<double>(x,y+1);
				double  d_y1= depth.at<double>(x+1, y);
				// calculate normal for each point
				Eigen::Matrix<double, 3,1> normal, v_x, v_y;
				v_x <<  ((d_x1-d)*(y-cx)+d_x1)/fx, (d_x1-d)*(x-cy)/fy , (d_x1-d);
				v_y << (d_y1-d)*(y-cx)/fx,(d_y1+ (d_y1-d)*(x-cy))/fy, (d_y1-d);
				v_x=v_x.normalized();
				v_y=v_y.normalized();
				normal=v_y.cross(v_x);// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111
//				normal=v_x.cross(v_y);
				normal=normal.normalized();

//				Vec3d d_n_rgb(normal.x(),normal.y(),normal.z());
				Vec3d d_n_rgb(normal.z()*0.5+0.5, normal.y()*0.5+0.5, normal.x()*0.5+0.5);
//				Vec3d d_n(normal.z(), normal.y(), normal.x());
				normalsMap_bgr.at<Vec3d>(x, y) = d_n_rgb;
//				normalsMap.at<Vec3d>(x, y) = d_n;
			}
		}

		// normal map filtering

//	    Mat normalMapafterfilter= normalMapFilter(normalsMap);// when checking result, uncommit it

//		Mat normalMapafterfilter_channel[3];
//		vector<Mat> channels_result;
//
//		split(normalMapafterfilter, normalMapafterfilter_channel);
//
//		Mat normal_zafterfilter= normalMapafterfilter_channel[0]*0.5+0.5;
//		Mat normal_yafterfilter= normalMapafterfilter_channel[1]*0.5+0.5;
//		Mat normal_xafterfilter= normalMapafterfilter_channel[2]*0.5+0.5;
//
//		channels_result.push_back(normal_zafterfilter);
//		channels_result.push_back(normal_yafterfilter);
//		channels_result.push_back(normal_xafterfilter);
//
//
//		Mat normalMapafterfilter_result;
//		merge(channels_result, normalMapafterfilter_result);
//
//
		imshow("normalsMap_1", normalsMap_bgr);
		waitKey(0);
		return normalMapFilter(normalsMap);


	}


	Mat getNormals_renderedDepth(const Eigen::Matrix<double,4,4> & M, const Mat& depth){

		Mat normalsMap(depth.rows, depth.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R
		Mat normalsMap_bgr(depth.rows, depth.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R
		double fov_y= 33.398;
		double near= 0.01;
		double far= 60.0;
		double aspect= 1.333;
		double coeA= 2*far*near/(near-far);
		double coeB= (far+near)/(near-far);
		double f= 1.0/(tan(0.5*fov_y)*aspect);


		for(int x = 0; x < depth.rows; ++x)
		{
			for(int y = 0; y < depth.cols; ++y)
			{

				double d_mapped= 2.0 *depth.at<double>(x,y)-1.0;
				double x_mapped= 2.0* x/ 480.0 -1.0;
				double y_mapped= 2.0* y/ 640.0 -1.0;

//				Eigen::Vector4d p_h(y_mapped,x_mapped,d_mapped,1.0);
//				Eigen::Vector4d projPointInCam= M.inverse()*p_h;
//				Eigen::Matrix<double,3,1> p_3d;
//				p_3d<< projPointInCam.x()/projPointInCam.w(),  projPointInCam.y()/projPointInCam.w(), projPointInCam.z()/projPointInCam.w();

				Eigen::Matrix<double,3,1> p_3d;
				p_3d.z()=-coeA/(d_mapped+coeB);
				p_3d.x()=y_mapped*aspect*(-p_3d.z())/f;
				p_3d.y()=x_mapped*(-p_3d.z())/f;

				double d_x1= 2.0 *depth.at<double>(x,y+1)-1.0;
				double y_1mapped= 2.0* (y+1)/ 640.0 -1.0;
//				Eigen::Vector4d p_h_1(y_1mapped, x_mapped,d_x1,1.0);
//				Eigen::Vector4d projPointInCam1= M.inverse()*p_h_1;
				Eigen::Matrix<double,3,1> p_3d1;
//				p_3d1<< projPointInCam1.x()/projPointInCam1.w(),  projPointInCam1.y()/projPointInCam1.w(), projPointInCam1.z()/projPointInCam1.w();
				p_3d1.z()=-coeA/(d_x1+coeB);
				p_3d1.x()=y_1mapped*aspect*(-p_3d1.z())/f;
				p_3d1.y()=x_mapped*(-p_3d1.z())/f;




				double  d_y1=  2.0 *depth.at<double>(x+1, y)-1.0;
				double x_1mapped= 2.0* (x+1)/ 480.0 -1.0;

//				Eigen::Vector4d p_h_2(y_mapped, x_1mapped,d_y1,1.0);
//				Eigen::Vector4d projPointInCam2= M.inverse()*p_h_1;
				Eigen::Matrix<double,3,1> p_3d2;
//				p_3d2<< projPointInCam2.x()/projPointInCam2.w(),  projPointInCam2.y()/projPointInCam2.w(), projPointInCam2.z()/projPointInCam2.w();

				p_3d2.z()=-coeA/(d_y1+coeB);
				p_3d2.x()=y_mapped*aspect*(-p_3d2.z())/f;
				p_3d2.y()=x_1mapped*(-p_3d2.z())/f;

				Eigen::Vector3d v_x(p_3d1-p_3d);
				Eigen::Vector3d v_y(p_3d2-p_3d);


				Eigen::Vector3d normal;
				v_x=v_x.normalized();
				v_y=v_y.normalized();
				normal=v_x.cross(v_y);
				normal=normal.normalized();


				Vec3d d_n_rgb(normal.z()*0.5+0.5, normal.y()*0.5+0.5, normal.x()*0.5+0.5);
//				Vec3d d_n(normal.z(), normal.y(), normal.x());
				normalsMap_bgr.at<Vec3d>(x, y) = d_n_rgb;
//				normalsMap.at<Vec3d>(x, y) = d_n;
			}
		}

		imshow("normalsMap2", normalsMap_bgr);
		waitKey(0);
		return normalMapFilter(normalsMap_bgr);


	}




	Mat deltaMapGT(Mat& Img_left,  Mat& depth_left,  Mat& Img_right, Mat& depth_right,const Eigen::Matrix<double,3,3> & K_, double & thres, const  Sophus::SE3d & ExtrinsicPose){

//		Mat normalMap_left=getNormals(K_,depth_left);
//		Mat normalMap_right=getNormals(K_,depth_right);

		Mat deltaMapGT(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(1.0)); // default value 1.0
		double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2);
		pcl::PCDWriter writer;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rig (new pcl::PointCloud<pcl::PointXYZ>);


		for(int x = 0; x < depth_left.rows; ++x) {
			for (int y = 0; y < depth_left.cols; ++y) {
				double d_r= depth_right.at<double>(x,y);
				Eigen::Matrix<double,3,1> p_3d_no_d_r;
				p_3d_no_d_r<< (y-cx)/fx, (x-cy)/fy,1.0;
				Eigen::Matrix<double, 3,1> p_c2=d_r*p_3d_no_d_r;
				cloud_rig->push_back(pcl::PointXYZ(p_c2.x(), p_c2.y(), p_c2.z()));
			}
		}

		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud (cloud_rig);
		double  max, min;
		cv::minMaxLoc(Img_left, &min, &max);
		cout<<"\n show max and min of Img_left:\n"<< max <<","<<min<<endl;


		cv::minMaxLoc(Img_right, &min, &max);
		cout<<"\n show max and min of Img_right:\n"<< max <<","<<min<<endl;


		std::unordered_map<int, int> inliers_filter;
		//new image

		inliers_filter.emplace(173,333); //yes
		inliers_filter.emplace(378,268); //yes

		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquaredDistance;

		float radius =thres;

		for(int x = 0; x < depth_left.rows; ++x)
		{
			for(int y = 0; y < depth_left.cols; ++y)
			{

//				if(inliers_filter.count(x)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//				if(inliers_filter[x]!=y ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~


				// calculate 3D point of left camera
				double d= depth_left.at<double>(x,y);
				Eigen::Matrix<double,3,1> p_3d_no_d;
				p_3d_no_d<< (y-cx)/fx, (x-cy)/fy,1.0;
				Eigen::Matrix<double, 3,1> p_c1=d*p_3d_no_d;

				Eigen::Vector3d  point_Trans=ExtrinsicPose.rotationMatrix() *  p_c1+ExtrinsicPose.translation();

				pcl::PointXYZ searchPoint(point_Trans.x(),point_Trans.y(),point_Trans.z());
				if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
				{
//					for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
						std::cout << "\n------"  <<   (*cloud_rig)[ pointIdxRadiusSearch[0] ].x
						          << " " << (*cloud_rig)[ pointIdxRadiusSearch[0] ].y
						          << " " << (*cloud_rig)[ pointIdxRadiusSearch[0] ].z
						          << " (squared distance: " << pointRadiusSquaredDistance[0] << ")" << std::endl;


					double left_intensity=Img_left.at<double>(x,y);
					float pointCorres_x= (*cloud_rig)[ pointIdxRadiusSearch[0] ].x;
					float pointCorres_y= (*cloud_rig)[ pointIdxRadiusSearch[0] ].y;
					float pointCorres_z= (*cloud_rig)[ pointIdxRadiusSearch[0] ].z;
					float pixel_x=(fx*pointCorres_x)/pointCorres_z+cx;
					float pixel_y= (fy*pointCorres_y)/pointCorres_z+cy;
					double right_intensity=Img_right.at<double>((int)pixel_y, (int)pixel_x);
					if ( left_intensity <0.01 || right_intensity<0.01){ continue;}

					if(left_intensity/right_intensity> 2.0 || left_intensity/right_intensity<0.5){ continue;}
					deltaMapGT.at<float>(x,y)=left_intensity/right_intensity;


				}


//				double left_intensity=Img_left.at<double>(x,y);
//				for(int x_ = 0; x_ < depth_left.rows; ++x_) {
//					for (int y_ = 0; y_ < depth_left.cols; ++y_) {
//
//						double d_r= depth_right.at<double>(x_,y_);
//						Eigen::Matrix<double,3,1> p_3d_no_d_r;
//						p_3d_no_d_r<< (y_-cx)/fx, (x_-cy)/fy,1.0;
//						Eigen::Matrix<double, 3,1> p_c2=d_r*p_3d_no_d_r;
//						cloud_rig->push_back(pcl::PointXYZ(p_c2.x(), p_c2.y(), p_c2.z()));
//						double distance= ((point_Trans-p_c2).norm());
//						if ( distance< thres){
//							double rigth_intensity=Img_right.at<double>(x_,y_);
//							deltaMapGT.at<double>(x,y)=left_intensity/rigth_intensity;
//						}
//					}
//				}


//				double d_r= depth_right.at<double>(x,y);
//				Eigen::Matrix<double,3,1> p_3d_no_d_r;
//				p_3d_no_d_r<< (y-cx)/fx, (x-cy)/fy,1.0;
//				Eigen::Matrix<double, 3,1> p_c2=d_r*p_3d_no_d_r;
//				cloud_rig->push_back(pcl::PointXYZ(p_c2.x(), p_c2.y(), p_c2.z()));

			}
		}

//		writer.write("PointCloud_Transformed.pcd",*cloud, false);// do we need the sensor acquisition origin?
//		writer.write("PointCloud_right.pcd",*cloud_rig, false);// do we need the sensor acquisition origin?

//		double  max, min;
		cv::minMaxLoc(deltaMapGT, &min, &max);

		deltaMapGT=deltaMapGT*(1.0/(2.0-0.5))+(-0.5*(1.0/(2-0.5)));
//		deltaMap=deltaMap*(1.0/(3.8044-0.356712))+(-0.356712*(1.0/(3.8044-0.356712)));

		cout<<"\n show max and min of deltaMapGT:\n"<< max <<","<<min<<endl;
		imshow("DeltaMapGT", deltaMapGT);
//		waitKey(0);

		return deltaMapGT;
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