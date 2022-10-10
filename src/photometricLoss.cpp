
// read image
// select pixel
// optimization
// pyramid improvement

//local header files
#include <reprojection.h>
#include <photometricBA.h>
#include "ultils.h"
#include "PCLOpt.h"

//#include <algorithm>
//#include <atomic>
//#include <chrono>
#include <iostream>

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

	Eigen::Matrix<double,3,1> light_source(7.1, -22.9 ,11.0);
	Eigen::Matrix<double,3,1> Camera1(3.8, -16.5 ,26.1);
	Eigen::Matrix<double,3,1> Camera2(0.3 ,-16.9, 27.7);

	Eigen::Matrix3d R1,R2, R12;
	Eigen::Vector3d l_w,N_w, t12, t1,t2;



	Eigen::Quaterniond q_1(0.009445649,-0.0003128,-0.9994076,-0.0330920); //  cam1  wxyz
	Eigen::Quaterniond q_2(-0.08078633,-0.0084485,-0.9962677,-0.0292077 ); //  cam5  wxyz
	t1<< 3.8, -16.5, 26.1;
	t2 << -5.1,-15.2 ,27.5 ;

	R1=q_1.toRotationMatrix();
	R2=q_2.toRotationMatrix();
//	cout<< "show R1:\n"<< R1<<endl;
//	cout<< "show R2:\n"<< R2<<endl;
    R12= R2.transpose() * R1;
	t12= R2.transpose()* (t1-t2);

	Eigen::Quaterniond q_12(R12);


//	cout<< "show R12:\n"<< R12<<endl;
//	cout<< "show t12:\n"<< t12<<endl;




	l_w<< 0.223529, 0.490196, 0.843137;
	N_w<< 0.0352942, -0.223529, 0.976471;


//	cout << "\n show normal in C1 \n"<<R1w.transpose()*N_w<<endl;






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



//	loaded images
//	string image_ref_path = "../data/rgb/1305031102.175304.png"; // data/rgb/1305031102.175304.png, data_test/rgb/1305031453.359684.png
//	string image_target_path = "../data/rgb/1305031102.275326.png";  // matlab 1305031102.175304
//	string depth_ref_path = "../data/depth/1305031102.160407.png";  //   matlab      1305031102.262886

    // old dataset
//	string image_ref_path = "../data/rgb/newviewpoint1_colorful.png";
//	string image_target_path = "../data/rgb/viewpoint4.png";
//	string depth_ref_path = "../data/depth/viewpoint1_depth.exr";

//	// new  dataset
	string image_ref_path = "../data/rgb/vp1.png";
	string image_target_path = "../data/rgb/vp5.png";

	// no texture
//	string image_ref_path = "../data/rgb/cam1_notexture.png";
//	string image_target_path = "../data/rgb/cam5_notexture.png";



//	string depth_ref_path = "../data/depth/old_lineareyedepth.exr";
//	string depth_ref_path = "../data/depth/new_cam1.exr";

	string depth_ref_path = "../data/depth/cam1_depth.exr";
	//	string depth_target_path = "../data/depth/old_cam5.exr";
   //	string depth_target_path = "../data/depth/new_cam5.exr";
	string depth_target_path = "../data/depth/cam5_depth.exr";



//	string BRDFVals = "../data/depth/rt_14_52_47_cam1_diffuse.exr";//       yes
	string BRDFVals = "../data/depth/rt_14_50_34_cam1_spec.exr";//          no!

//	string BRDFVals = "../data/depth/rt_15_3_19_cam1_spdistribution.exr";// no!   but    yes for another pixel
//	string BRDFVals = "../data/depth/rt_14_55_15_cam1_fresnel.exr";//  yes!,, yes
//	string BRDFVals = "../data/depth/rt_14_56_47_cam1_geo.exr";//  almost  yes

//	string BRDFVals = "../data/depth/rt_18_16_4_cam1_specColor.exr";//
//	string BRDFVals = "../data/depth/rt_18_21_4_cam1_ggx.exr";//
	string NdotH_GT_Map_path = "../data/depth/rt_20_23_47_cam1_NdotH.exr";//
	string NdotH_GT_Map5_path = "../data/depth/rt_20_23_47_cam5_NdotH.exr";//




	string NdotL_GT_Map_path = "../data/depth/rt_21_54_36_cam1_NdotL.exr";//
	string NdotV_GT_Map_path = "../data/depth/rt_21_55_49_cam1_NdotV.exr";//
	string NdotV_GT_Map5_path = "../data/depth/rt_21_55_49_cam5_NdotV.exr";//


//	Eigen::Vector2i pixel_pos(173,333);
//	Eigen::Vector2i pixel_pos(378,268);
	Eigen::Vector2i pixel_pos(213,295);
//	Eigen::Vector2i pixel_pos(370,488);


//	imageInfo(BRDFVals, pixel_pos);












	//data/rgb/viewpoint1_mr.png
	// read metallic adn roughness data, read metallic adn roughness data
	string image_ref_MR_path = "../data/rgb/vp1_mr.png"; // store value in rgb channels,  channel b: metallic, channel green: roughness
	string image_target_MR_path = "../data/rgb/vp5_mr.png";






	//  create a metallic and roughness table, (map R G values into [0 ,1] for two images)
	// create a metallic and roughness table for reference image
	Mat image_ref_MR= imread(image_ref_MR_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    Mat ref_mr_table[3];
	split(image_ref_MR,ref_mr_table);// 0: red, 1: green, 2: blue
	Mat image_ref_metallic=  ref_mr_table[2];
	Mat image_ref_roughness= ref_mr_table[1];

	image_ref_metallic.convertTo(image_ref_metallic, CV_32FC1,1.0 / 255.0);

	image_ref_roughness.convertTo(image_ref_roughness, CV_32FC1,1.0 / 255.0);
//	imshow("image_ref_roughness", image_ref_roughness);
//	waitKey(0);
    //  create a metallic and roughness table for target image
	Mat image_target_MR= imread(image_target_MR_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	Mat taget_mr_table[3];
	split(image_target_MR,taget_mr_table);// 0: red, 1: green, 2: blue
	Mat image_target_metallic=  taget_mr_table[2];
	Mat image_target_roughness= taget_mr_table[1];
	image_target_metallic.convertTo(image_target_metallic, CV_64FC1,1.0 / 255.0);
	image_target_roughness.convertTo(image_target_roughness, CV_64FC1,1.0 / 255.0);

//	imshow("image_ref_metallic",image_ref_metallic);
//	imshow("image_ref_roughness",image_ref_roughness);
//	waitKey(0);





	// read base color data TODO: check if we need to map the value of baseColor
	string image_ref_baseColor_path = "../data/rgb/vp1_basecolor.png";
	string image_target_baseColor = "../data/rgb/vp5_basecolor.png";

	// no texture
//	string image_ref_baseColor_path = "../data/rgb/cam1_color.png";
//	string image_target_baseColor = "../data/rgb/cam5_color.png";



	Mat image_ref_baseColor= imread(image_ref_baseColor_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	Mat image_right_baseColor= imread(image_target_baseColor,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	image_ref_baseColor.convertTo(image_ref_baseColor, CV_32FC3, 1.0/255.0);
	image_right_baseColor.convertTo(image_right_baseColor,CV_32FC3,1.0/255.0);
	double min, max;
	cv::minMaxIdx(image_ref_baseColor, &min, &max);
	cout<<"show the depth_ref value range"<<"min:"<<min<<"max:"<<max<<endl;
//	imshow("image_ref_baseColor",image_ref_baseColor);
//	imshow("grayImage_target",grayImage_target);
//	waitKey(0);



	//	string image_ref_path = "../data_test/rgb/1305031117.243277.png"; //  , data_test/rgb/1305031453.359684                data_test/rgb/1305031453.359684.png
	//	string image_target_path = "../data_test/rgb/1305031117.843291.png";  // matlab 1305031102.175304
	//	string depth_ref_path = "../data_test/depth/1305031117.241340.png";  //   matlab      1305031102.262886


	Mat grayImage_target, grayImage_ref;
	// read target image
	Mat image_target = imread(image_target_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	// color space conversion
//	cvtColor(image_target, grayImage_target, COLOR_BGR2GRAY);   right


	Mat tar_ch_red;
	int channelIdx= 1;
	extractChannel(image_target, tar_ch_red, channelIdx);
	grayImage_target=tar_ch_red;


	

	// precision improvement
	grayImage_target.convertTo(grayImage_target, CV_64FC1, 1.0 / 255.0);
	// read ref image
	Mat image_ref = imread(image_ref_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	// color space conversion
//	cvtColor(image_ref, grayImage_ref, COLOR_BGR2GRAY);   // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11 left
    Mat ref_ch_red;

	extractChannel(image_ref, ref_ch_red, channelIdx);
	grayImage_ref=ref_ch_red;


	imshow("grayImage_ref",grayImage_ref);
	imshow("grayImage_target",grayImage_target);
	waitKey(0);





	// precision improvement
	grayImage_ref.convertTo(grayImage_ref, CV_64FC1, 1.0 / 255.0);
	Mat depth_ref = imread(depth_ref_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	Mat depth_target = imread(depth_target_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

	// left map depth
	Mat channel[3],depth_ref_render, channel_tar[3], depth_tar_render;
	split(depth_ref,channel);
	depth_ref=channel[0];
//	imageInfo(depth_ref_path,pixel_pos);



//	depth_ref.convertTo(depth_ref_render, CV_64FC1);
//	depth_ref= depth_ref_render *(60.0-0.01) + 0.01;
	depth_ref.convertTo(depth_ref, CV_64FC1);// ?????????????????????????????????????????????

//		cv::minMaxIdx(depth_ref, &min, &max);
//		cout<<"\n show the depth_ref value range:\n"<<"min:"<<min<<"max:"<<max<<endl;

//		cout<<"depth of depth_ref"<<depth_ref.depth()<<"!!!!!!!!!!!!!!"<<endl;


   cv::minMaxIdx(depth_ref, &min, &max);
   cout<<"\n show the depth_ref value range:\n"<<"min:"<<min<<"max:"<<max<<endl;

   // right map depth
	split(depth_target, channel_tar);
	depth_target=channel_tar[0];
//	depth_target.convertTo(depth_tar_render,CV_64FC1);
//	depth_target=depth_tar_render *(60.0-0.01) + 0.01;
	depth_target.convertTo(depth_target, CV_64FC1);




//   depth_target.convertTo(depth_target, CV_64F);
//   depth_target = depth_target / 5000.0;


	// 相机内参
	Eigen::Matrix3d K;
	Eigen::Matrix4d M;


	double fov_y= 33.398;
	double near= 0.01;
	double far= 60.0;
	double aspect= 1.333;


//	K << 517.3, 0, 318.6,
//			0, 516.5, 255.3,
//			0, 0, 1;
//	double cx = 325.5;
//	double cy = 253.5;
//	double fx = 518.0;
//	double fy = 519.0;

//	K<<fx,0,cx,0,fy,cy,0,0,1.0;

// K<< 800.0, 0, 0,
//     0, 800.0, 0,
//	 0,   0,  1;
	K<< 800.0, 0, 320,
	    0, 800.0, 240,
		0,   0,  1;

 M << 1.0/(tan(0.5*fov_y)*aspect), 0, 0, 0,
      0,  atan(0.5*fov_y), 0   ,  0,
	  0,0, (far+near)/(near-far), 2*far*near/(near-far),
	  0,  0,   -1,    0;



// TODO: test the current pose and compare it with the one using function
//	Mat normalMap_=getNormals(K,depth_ref);
//	imshow("normalsMap_1", normalMap_);
//	Mat normalMap2= getNormals_renderedDepth(M, depth_ref_render);

//	Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic> eigenMat;
//	cv2eigen(normalMap,eigenMat);

//	cout<<"show Eigenmat image properties:"<< eigenMat.size()<<endl;

	Sophus::SE3d xi;
	Eigen::Matrix<double, 3,3> R;
	Eigen::Matrix<double, 3,1> t;

//	R<< 0.9990,  0.0210 ,   0.0395,
//	   -0.0219 ,   0.9995 ,  0.0237,
//	    -0.0389,   -0.0246,  0.9989;
//R << 0.9998,   -0.0031  ,  0.0183,
//	0.0030  ,  1.0000 ,   0.0059,
//	-0.0183 ,  -0.0058,    0.9998;


//	R = 0.7202, 0.2168, 0.6591,
//	        -0.4602,  0.8601,  0.2200,
//			-0.5192,  -0.4617, 0.7192;
// GT
//  Eigen::Quaterniond q(1.0000  ,  0.0092 ,   0.0029    ,0.0015); // ( 1.0000  ,  0.0016,    0.0093  , -0.0017);
//	Eigen::Quaterniond q(0.9998,    0.0174 ,   0.0076 ,  -0.0003);
	// disturbing rotation
//	Eigen::Quaterniond q( 0.9998,    0.0174 ,   0.0076 ,  -0.0003);
    //	0.9949 ,  -0.0103  ,  0.0206 ,  -0.0978 : 10 degree disturbing only in one axis. yes
	// 0.9913    0.0811   -0.0597    0.0843  :10 degree disturbing only in each axis.
	// 0.9986    0.0329   -0.0221    0.0351 : 5 degree in each axis,

	R=q_12.normalized().toRotationMatrix();
// GT
//t <<-3.4990,
//	0.3043,
//	1.6231;

t <<  3.5266,
      -0.1558,
	  1.5840;
//	t<<   0.0028,
//	      -0.0053,
//	      -0.0362;
	// disturbing translation
//	t<<  0.0223,
//	     0.1023,
//		 -0.0246;

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
	Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K,distanceThres,xi);
//	double mingt, maxgt;
//	cv::minMaxLoc(deltaMapGT_res, &mingt, &maxgt);
//	cout<<"show the deltaMapGT_res value range"<<"min:"<<mingt<<"max:"<<maxgt<<endl;

	std::vector<Eigen::Vector3d> pts;
	cv::Mat normal_map(480, 640, CV_64FC3);
	double fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy = K(1, 2), f=30.0;

	for (int u = 0; u< depth_ref.rows; u++) // colId, cols: 0 to 480
	{
		for (int v = 0; v < depth_ref.cols; v++) // rowId,  rows: 0 to 640
		{

			double d=depth_ref.at<double>(u,v);
			double d_x1= depth_ref.at<double>(u,v+1);
			double d_y1= depth_ref.at<double>(u+1, v);

			// calculate 3D point coordinate
			Eigen::Vector2d pixelCoord((double)v,(double)u);//  u is the row id , v is col id
			Eigen::Vector3d p_3d_no_d((pixelCoord(0)-cx)/fx, (pixelCoord(1)-cy)/fy,1.0);
			Eigen::Vector3d p_c1=d*p_3d_no_d;

			pts.push_back(p_c1);
			Eigen::Matrix<double,3,1> normal, v_x, v_y;
			v_x <<  ((d_x1-d)*(v-cx)+d_x1)/fx, (d_x1-d)*(u-cy)/fy , (d_x1-d);
			v_y << (d_y1-d)*(v-cx)/fx,(d_y1+ (d_y1-d)*(u-cy))/fy, (d_y1-d);
			v_x=v_x.normalized();
			v_y=v_y.normalized();
            normal=v_y.cross(v_x);
//			normal=v_x.cross(v_y);
			normal=normal.normalized();

			normal_map.at<cv::Vec3d>(u, v)[0] = normal(0);
			normal_map.at<cv::Vec3d>(u, v)[1] = normal(1);
			normal_map.at<cv::Vec3d>(u, v)[2] = normal(2);

		}
	}
	comp_accurate_normals(pts, normal_map);
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
		Mat deltaMap(depth_ref.rows, depth_ref.cols, CV_32FC1, Scalar(1)); // storing delta

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








        int i=0;
		while ( i < 2){
//			PhotometricBA(IRef, I, options, Klvl, xi, DRef,deltaMap);
			updateDelta(xi,Klvl,image_ref_baseColor,DRef,image_ref_metallic ,image_ref_roughness,light_source,deltaMap,newNormalMap);// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


//			deltaMap.convertTo(deltaMap, CV_32FC1, 200.0);
//			imshow("deltamap", deltaMap);
//			waitKey(0);

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
			Mat mask = cv::Mat(deltaMap != deltaMap);

			deltaMap.setTo(1.0, mask);

//			Mat mask_specular = cv::Mat( (deltaMap < 0.85) && (deltaMap > 1.15));
//			Mat mask_specular_2 = cv::Mat(deltaMap < 0.85);
			double  max_n, min_n;
			cv::minMaxIdx(deltaMap, &min_n, &max_n);
			cout<<"show max and min"<< max_n <<","<<min_n<<endl;

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

			Mat result, result2;
//			grayImage_ref.copyTo(result,mask_specular);
//			grayImage_ref.copyTo(result2,mask_specular_2 );
//			deltaMap=deltaMap*(1.0/(3.8044-0.356712))+(-0.356712*(1.0/(3.8044-0.356712)));

			deltaMap=deltaMap*(1.0/(max_n-min_n))+(-min_n*(1.0/(max_n-min_n)));
			imshow("show masked deltaMap", deltaMap);
//			Mat BigDelta;
//			deltaMap.convertTo(BigDelta, CV_8UC1, 255.0);
//
//			imshow("BigDelta",BigDelta);
//			imwrite("notexture.",deltaMap);

//			imshow("show masked image2", result2);
			waitKey(0);





          i+=1;

		}




//		cout<<"optimized test value: "<<DRef.at<double>(363,376)<<endl;
		cout << "\n Show optimized pose:\n" << xi.rotationMatrix() << "\n Show translation:\n" << xi.translation()
		     << endl;
		Eigen::Quaterniond q_opt( xi.rotationMatrix());
		cout<<"\n Show the optimized rotation as quaternion:"<<q_opt.w()<<","<<q_opt.x()<<","<<q_opt.y()<<","<<q_opt.z()<< endl;

	}

	return 0;
}

