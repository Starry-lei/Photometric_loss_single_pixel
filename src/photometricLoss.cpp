
// read image
// select pixel
// optimization
// pyramid improvement

//local header files
#include <reprojection.h>
#include <photometricBA.h>
#include "ultils.h"
#include "PCLOpt.h"
#include "dataLoader.h"

//#include <algorithm>
//#include <atomic>
//#include <chrono>

#include <sophus/se3.hpp>
//#include <tbb/concurrent_unordered_map.h>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
//#include "opencv2/features2d.hpp"



#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>
#include <visnav/local_parameterization_se3.hpp>

using namespace cv;
using namespace std;
using namespace DSONL;



int main(int argc, char **argv) {




//	Eigen::Vector2i pixel_pos(173,333);
//	Eigen::Vector2i pixel_pos(378,268);
	Eigen::Vector2i pixel_pos(213,295);
//	Eigen::Vector2i pixel_pos(370,488);


  //  create a metallic and roughness table, (map R G values into [0 ,1] for two images)
//	// create a metallic and roughness table for reference image
//	Mat image_ref_MR= imread(image_ref_MR_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    Mat ref_mr_table[3];
	split(image_ref_MR,ref_mr_table);// 0: red, 1: green, 2: blue
	Mat image_ref_metallic=  ref_mr_table[2];
	Mat image_ref_roughness= ref_mr_table[1];


	image_ref_metallic.convertTo(image_ref_metallic, CV_32FC1,1.0 / 255.0);
	image_ref_roughness.convertTo(image_ref_roughness, CV_32FC1,1.0 / 255.0);
// no need to convertTO
//	image_ref_metallic.convertTo(image_ref_metallic, CV_32FC1,1.0 / 255.0);
//	image_ref_roughness.convertTo(image_ref_roughness, CV_32FC1,1.0 / 255.0);


//	imshow("image_ref_roughness", image_ref_roughness);
//	waitKey(0);

    //  create a metallic and roughness table for target image
//	Mat image_target_MR= imread(image_target_MR_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	Mat taget_mr_table[3];
	split(image_target_MR,taget_mr_table);// 0: red, 1: green, 2: blue
	Mat image_target_metallic=  taget_mr_table[2];
	Mat image_target_roughness= taget_mr_table[1];
	image_target_metallic.convertTo(image_target_metallic, CV_64FC1,1.0 / 255.0);
	image_target_roughness.convertTo(image_target_roughness, CV_64FC1,1.0 / 255.0);
//	image_target_metallic.convertTo(image_target_metallic, CV_64FC1);
//	image_target_roughness.convertTo(image_target_roughness, CV_64FC1);
//	imshow("image_ref_metallic",image_ref_metallic);
//	imshow("image_ref_roughness",image_ref_roughness);
//	imageInfo(image_target_MR_path, pixel_pos);
//	waitKey(0);

	Mat grayImage_target, grayImage_ref;
	Mat tar_ch_red;
	int channelIdx= 1;
	extractChannel(image_target, tar_ch_red, channelIdx);
	grayImage_target=tar_ch_red;

    Mat ref_ch_red;
	extractChannel(image_ref, ref_ch_red, channelIdx);
	grayImage_ref=ref_ch_red;

//
	grayImage_ref.convertTo(grayImage_ref,CV_64FC1);
	grayImage_target.convertTo(grayImage_target, CV_64FC1);
//	imshow("grayImage_ref",grayImage_ref);
//	imshow("grayImage_target",grayImage_target);




//	Mat depth_ref = imread(depth_ref_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
//	Mat depth_target = imread(depth_target_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

	// left map depth
	Mat channel[3],depth_ref_render, channel_tar[3], depth_tar_render;
	split(depth_ref,channel);
	depth_ref=channel[0];
	depth_ref.convertTo(depth_ref, CV_64FC1);
   // right map depth
	split(depth_target, channel_tar);
	depth_target=channel_tar[0];
	depth_target.convertTo(depth_target, CV_64FC1);



	K<< 800.0, 0, 320,
	    0, 800.0, 240,
		0,   0,  1;

//	K<< 3200.0, 0, 1280.0,
//	    0, 3200.0, 720.0,
//		0,   0,  1;

    M << 1.0/(tan(0.5*fov_y)*aspect), 0, 0, 0,
      0,  atan(0.5*fov_y), 0   ,  0,
	  0,0, (far+near)/(near-far), 2*far*near/(near-far),
	  0,  0,   -1,    0;

//	Mat normalMap_=getNormals(K,depth_ref);
//	imshow("normalsMap_1", normalMap_);
//	Mat normalMap2= getNormals_renderedDepth(M, depth_ref_render);

	Sophus::SE3d xi;
	Eigen::Matrix<double, 3,3> R;
	Eigen::Matrix<double, 3,1> t;
	R=q_12.normalized().toRotationMatrix();
	t <<  3.5266,
	      -0.1558,
		  1.5840;


	xi.setRotationMatrix(R);
	xi.translation()=t12;
	cout << "\n Show initial pose:\n" << xi.rotationMatrix() << "\n Show translation:\n" << xi.translation()<<endl;


//	imshow("grayImage_ref",grayImage_ref);
//	imshow("grayImage_target",grayImage_target);
//	double min_in,max_in;
//	cv::minMaxLoc(grayImage_target, &min_in, &max_in);
//	cout<<"\n show min and max of grayImage_target:\n"<< min_in <<","<<max_in<<endl;
//	waitKey(0);




		double distanceThres=0.07;

	// HD distanceThres
//	double distanceThres=0.14;

	float upper=5; // 2.09998gt   // 0.335897
	float buttom=0.2;
	//	Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K,distanceThres,xi, upper, buttom);
	//	double mingt, maxgt;
	//	cv::minMaxLoc(deltaMapGT_res, &mingt, &maxgt);
	//	cout<<"show the deltaMapGT_res value range"<<"min:"<<mingt<<"max:"<<maxgt<<endl;
// ------------------------------------------------------------------------------------------Movingleast algorithm---------------------------------------------------------------
	std::vector<Eigen::Vector3d> pts;
	cv::Mat normal_map(depth_ref.rows, depth_ref.cols, CV_64FC3);
	double fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy = K(1, 2), f=30.0;
//
//	for (int u = 0; u< depth_ref.rows; u++) // colId, cols: 0 to 480
//	{
//		for (int v = 0; v < depth_ref.cols; v++) // rowId,  rows: 0 to 640
//		{
//
//			double d=depth_ref.at<double>(u,v);
//			double d_x1= depth_ref.at<double>(u,v+1);
//			double d_y1= depth_ref.at<double>(u+1, v);
//
//			// calculate 3D point coordinate
//			Eigen::Vector2d pixelCoord((double)v,(double)u);//  u is the row id , v is col id
//			Eigen::Vector3d p_3d_no_d((pixelCoord(0)-cx)/fx, (pixelCoord(1)-cy)/fy,1.0);
//			Eigen::Vector3d p_c1=d*p_3d_no_d;
//
//			pts.push_back(p_c1);
//			Eigen::Matrix<double,3,1> normal, v_x, v_y;
//			v_x <<  ((d_x1-d)*(v-cx)+d_x1)/fx, (d_x1-d)*(u-cy)/fy , (d_x1-d);
//			v_y << (d_y1-d)*(v-cx)/fx,(d_y1+ (d_y1-d)*(u-cy))/fy, (d_y1-d);
//			v_x=v_x.normalized();
//			v_y=v_y.normalized();
//            normal=v_y.cross(v_x);
////			normal=v_x.cross(v_y);
//			normal=normal.normalized();
//
//			normal_map.at<cv::Vec3d>(u, v)[0] = normal(0);
//			normal_map.at<cv::Vec3d>(u, v)[1] = normal(1);
//			normal_map.at<cv::Vec3d>(u, v)[2] = normal(2);
//
//		}
//	}
//	comp_accurate_normals(pts, normal_map);

//	normal_map_GT
//

//	imageInfo(normal_GT_path, pixel_pos); // normalized already





	for (int u = 0; u< depth_ref.rows; u++) // colId, cols: 0 to 480
	{
		for (int v = 0; v < depth_ref.cols; v++) // rowId,  rows: 0 to 640
		{

			Eigen::Vector3d normal_new( normal_map_GT.at<Vec3f>(u,v)[2],  -normal_map_GT.at<Vec3f>(u,v)[1], normal_map_GT.at<Vec3f>(u,v)[0]);// !!!!!!!!!!!!!!!!!!!!!!!

			normal_new= R1.transpose()*normal_new;

			Eigen::Vector3d principal_axis(0, 0, 1);
			if(normal_new.dot(principal_axis)>0)
			{
				normal_new = -normal_new;
			}

			normal_map.at<Vec3d>(u,v)[0]=normal_new(0);
			normal_map.at<Vec3d>(u,v)[1]=normal_new(1);
			normal_map.at<Vec3d>(u,v)[2]=normal_new(2);

		}
	}




// filtered normal map
//	cv::Mat normal_map_sdf(480, 640, CV_32FC3);
//	cv::Mat nx, ny, nz, med_depth;
//	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
//	NormalEstimator<float>* NEst;
//	NEst = new NormalEstimator<float>(640, 480, K, cv::Size(2*5+1, 2*5+1));
////	NEst = new NormalEstimator<float>(640, 480, K_, cv::Size(2*3+1, 2*3+1));
//
//	NEst->compute(depth_ref,nx, ny, nz);
//	cv::Mat* x0_ptr = NEst->x0_ptr();
//	cv::Mat* y0_ptr = NEst->y0_ptr();
//	cv::Mat* n_sq_inv_ptr = NEst->n_sq_inv_ptr();
//	const float* nx_ptr = (const float*)nx.data;
//	const float* ny_ptr = (const float*)ny.data;
//	const float* nz_ptr = (const float*)nz.data;
//
//	const float* x_hom_ptr = (const float*)x0_ptr->data;
//	const float* y_hom_ptr = (const float*)y0_ptr->data;
//
//	const float* hom_inv_ptr = (const float*)n_sq_inv_ptr->data;
//	const float* z_ptr = (const float*)depth_ref.data;
//	const float* zm_ptr = (const float*)med_depth.data;
//
//	for (int u = 0; u< depth_ref.rows; u++) // colId, cols: 0 to 480
//	{
//		for (int v = 0; v < depth_ref.cols; v++) // rowId,  rows: 0 to 640
//		{
//			const size_t idx = u * depth_ref.cols + v;
//			const float z = z_ptr[idx];
//
////	        if (z <= 0.5 || z >= 65.0 ) // z out of range or unreliable z
////		        continue;
//
//			const Eigen::Vector3f  xy_hom(x_hom_ptr[idx], y_hom_ptr[idx], 1.);
//			const Eigen::Vector3f normal(nx_ptr[idx], ny_ptr[idx], nz_ptr[idx]);
//			if (normal.squaredNorm() < .1) {continue; }
//			if (normal.dot(xy_hom) * normal.dot(xy_hom) * hom_inv_ptr[idx] < .25) // normal direction too far from viewing ray direction (>72.5°)
//				continue;
//
////	        Vec3f d_n_rgb( normal.normalized().z()*0.5+0.5,  normal.normalized().y()*0.5+0.5,  normal.normalized().x()*0.5+0.5);
//	        Vec3f d_n_rgb( normal.normalized().z(),  normal.normalized().y(),  normal.normalized().x());
////			Vec3f d_n_rgb( normal.z(),  normal.y(),  normal.x());
//			normal_map_sdf.at<Vec3f>(u,v)=d_n_rgb;
//
//		}
//	}
//	std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//	std::chrono::duration<double> time_used =
//			std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
//	cout << "construct the normals: " << time_used.count() << " seconds." << endl;
//
////	imshow("newNormalmap_sdf", normal_map_sdf);
////
////	waitKey(0);
//
//	for (int u = 0; u< depth_ref.rows; u++) // colId, cols: 0 to 480
//	{
//		for (int v = 0; v < depth_ref.cols; v++) // rowId,  rows: 0 to 640
//		{
//
//			Eigen::Vector3d normal_new( normal_map_sdf.at<Vec3f>(u,v)[2],  normal_map_sdf.at<Vec3f>(u,v)[1], normal_map_sdf.at<Vec3f>(u,v)[0]);
//
//			Eigen::Vector3d principal_axis(0, 0, 1);
//			if(normal_new.dot(principal_axis)>0)
//			{
//				normal_new = -normal_new;
//			}
//
//			normal_map.at<Vec3d>(u,v)[0]=normal_new(0);
//			normal_map.at<Vec3d>(u,v)[1]=normal_new(1);
//			normal_map.at<Vec3d>(u,v)[2]=normal_new(2);
//
//		}
//	}
//


	Mat newNormalMap=normal_map;


	int lvl_target, lvl_ref;
	for (int lvl = 1; lvl >= 1; lvl--)
	{
		cout << "\n Show the value of lvl:" << lvl << endl;
		Mat IRef, DRef, I, D;
		Eigen::Matrix3d Klvl, Klvl_ignore;
		lvl_target = lvl;
		lvl_ref = lvl;
		downscale(grayImage_ref, depth_ref, K, lvl_ref, IRef, DRef, Klvl);
		downscale(grayImage_target, depth_target, K, lvl_target, I, D, Klvl_ignore);
//	    imshow("Image_IRef", IRef);
//	    imshow("Image_I", I);
//	    waitKey(0);
		// float errLast = 1e-10;
		// float lambda = 0.1;
//	    double min2, max2;
//	    cv::minMaxIdx(DRef, &min2, &max2);
//	    cout<<"show the depth_ref value range"<<"min:"<<min2<<"max:"<<max2<<endl;
//         imshow("Image1", DRef);
//         waitKey(0);
//	    cv::Mat flat= I.reshape(1, I.total()*I.channels());
//	    std::vector<double> img_gray_values= I.isContinuous()? flat : flat.clone();
//	    printAll(&img_gray_values[0], img_gray_values.size());//~~~~~~~~~~~~~~~~~~


		PhotometricBAOptions options;

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


		Mat deltaMap(depth_ref.rows, depth_ref.cols, CV_32FC1, Scalar(1)); // storing delta
//		Mat deltaMap(depth_ref.rows, depth_ref.cols, CV_32FC3, Scalar(0)); // storing delta
        int i=0;
		while ( i < 2){
//			PhotometricBA(IRef, I, options, Klvl, xi, DRef,deltaMap);
			float up_new=upper;
			float butt_new=buttom;
			updateDelta(xi,Klvl,image_ref_baseColor,DRef,image_ref_metallic ,image_ref_roughness,light_source,deltaMap,newNormalMap,up_new, butt_new);// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

			Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K,distanceThres,xi, upper, buttom, deltaMap);
			Mat showGTdeltaMap=colorMap(deltaMapGT_res, upper, buttom);
			Mat showESdeltaMap=colorMap(deltaMap, upper, buttom);
			imshow("show GT deltaMap", showGTdeltaMap);
			imshow("show ES deltaMap", showESdeltaMap);
			imwrite("GT_deltaMap.tiff",showGTdeltaMap);
			imwrite("ES_deltaMap.tiff",showESdeltaMap);
			waitKey(0);
          i+=1;

		}

		cout << "\n Show optimized pose:\n" << xi.rotationMatrix() << "\n Show translation:\n" << xi.translation()
		     << endl;
		Eigen::Quaterniond q_opt( xi.rotationMatrix());
		cout<<"\n Show the optimized rotation as quaternion:"<<q_opt.w()<<","<<q_opt.x()<<","<<q_opt.y()<<","<<q_opt.z()<< endl;

	}

	return 0;
}






//			for(int x = 0; x < deltaMap.rows; ++x)
//			{
//				for(int y = 0; y < deltaMap.cols; ++y)
//
//					if (deltaMap.at<float>(x,y) <0.85 || deltaMap.at<float>(x,y)>1.15 ){
//						deltaMap.at<float>(x,y)=255;
//					}
//				else{
//						deltaMap.at<float>(x,y)=0;
//				}
//			}



//			for(int x = 0; x < deltaMap.rows; ++x)
//			{
//				for(int y = 0; y < deltaMap.cols; ++y)
//				{
////					if(inliers_filter.count(x)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
////					if(inliers_filter[x]!=y ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
////					if (deltaMap.at<float>(x,y)==-nan){cout << "coord"<< }
//					cout<<"show delta:"<<deltaMap.at<float>(x,y)<<endl;
//					int tst=1;
//
//
//				}
//			}