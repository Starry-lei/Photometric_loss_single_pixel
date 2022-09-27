
// read image
// select pixel
// optimization
// pyramid improvement

//local header files
#include <reprojection.h>
#include <photometricBA.h>
#include <ultils.h>


//#include <algorithm>
//#include <atomic>
//#include <chrono>
#include <iostream>

#include <sophus/se3.hpp>
//#include <tbb/concurrent_unordered_map.h>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
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

	Eigen::Matrix<double,3,1> light_source(-0.666, 0.1 ,-0.186);
	Eigen::Matrix<double,3,1> Camera1(3.8, -16.5 ,26.1);
	Eigen::Matrix<double,3,1> Camera2(0.3 ,-16.9, 27.7);

	Eigen::Matrix3d R1w;
//	Eigen::Matrix3d R_X=  rotmatx(-3.793*DEG_TO_ARC);
//	Eigen::Matrix3d R_Y=  rotmaty(-178.917*DEG_TO_ARC);
//	Eigen::Matrix3d R_Z=  rotmatz(0*DEG_TO_ARC);
//	Eigen::Matrix3d R_1w=  R_Y*R_X*R_Z;
	Eigen::Quaterniond q_1( 9.44564042e-03, -9.99407604e-01, -3.30926474e-02, -3.12766530e-04); // wxyz

	R1w=q_1.toRotationMatrix();

	Eigen::Vector3d l_w,N_w;
	l_w<< 0.223529, 0.490196, 0.843137;
	N_w<< 0.0352942, -0.223529, 0.976471;


	cout << "\n show normal in C1 \n"<<R1w.transpose()*N_w<<endl;






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
	string image_ref_path = "../data/rgb/newviewpoint1_colorful.png";
	string image_target_path = "../data/rgb/newviewpoint2_colorful.png";
	string depth_ref_path = "../data/depth/viewpoint1_depth.exr";
    //data/rgb/viewpoint1_mr.png
	// read metallic adn roughness data, read metallic adn roughness data
	string image_ref_MR_path = "../data/rgb/viewpoint1_mr.png"; // store value in rgb channels,  channel b: metallic, channel green: roughness
	string image_target_MR_path = "../data/rgb/viewpoint2_mr.png";
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






	// read base color data TODO: check if we need to map the value of baseColor
	string image_ref_baseColor_path = "../data/rgb/newviewpoint1_texture.png";
	string image_target_baseColor = "../data/rgb/newviewpoint2_texture.png";
	Mat image_ref_baseColor= imread(image_ref_baseColor_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

	image_ref_baseColor.convertTo(image_ref_baseColor, CV_32FC3, 1.0/255.0);
	double min, max;
	cv::minMaxIdx(image_ref_baseColor, &min, &max);
	cout<<"show the depth_ref value range"<<"min:"<<min<<"max:"<<max<<endl;












	//	string image_ref_path = "../data_test/rgb/1305031117.243277.png"; //  , data_test/rgb/1305031453.359684                data_test/rgb/1305031453.359684.png
	//	string image_target_path = "../data_test/rgb/1305031117.843291.png";  // matlab 1305031102.175304
	//	string depth_ref_path = "../data_test/depth/1305031117.241340.png";  //   matlab      1305031102.262886

	string depth_target_path = "../data/depth/1305031102.160407.png";
	Mat grayImage_target, grayImage_ref;
	// read target image
	Mat image_target = imread(image_target_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	// color space conversion
//	cvtColor(image_target, grayImage_target, COLOR_BGR2GRAY);  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! right


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


//	imshow("grayImage_ref",grayImage_ref);
//	imshow("grayImage_target",grayImage_target);
//	waitKey(0);





	// precision improvement
	grayImage_ref.convertTo(grayImage_ref, CV_64FC1, 1.0 / 255.0);
	Mat depth_ref = imread(depth_ref_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
//	cout<<"show the depth() of image:\n "<<depth_ref.depth()<<"and channels:"<<depth_ref.channels()<<endl;
	Mat depth_target = imread(depth_target_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	Mat channel[3],depth_ref_render;
	split(depth_ref,channel);
	depth_ref=channel[0];


	depth_ref.convertTo(depth_ref_render, CV_64FC1);

	depth_ref= depth_ref *(60.0-0.01) + 0.01;
	depth_ref.convertTo(depth_ref, CV_64FC1);

   cv::minMaxIdx(depth_ref, &min, &max);
   cout<<"show the depth_ref value range"<<"min:"<<min<<"max:"<<max<<endl;


//   depth_target.convertTo(depth_target, CV_64F);
//   depth_target = depth_target / 5000.0;


//	double min, max;
//	 cv::minMaxIdx(grayImage_ref, &min, &max);
//	 cout<<"show the depth_ref value range"<<"min:"<<min<<"max:"<<max<<endl;

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
//	Mat normalMap=getNormals(K,depth_ref);
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
  Eigen::Quaterniond q(1.0000  ,  0.0092 ,   0.0029    ,0.0015); // ( 1.0000  ,  0.0016,    0.0093  , -0.0017);
//	Eigen::Quaterniond q(0.9998,    0.0174 ,   0.0076 ,  -0.0003);
	// disturbing rotation
//	Eigen::Quaterniond q( 0.9998,    0.0174 ,   0.0076 ,  -0.0003);
    //	0.9949 ,  -0.0103  ,  0.0206 ,  -0.0978 : 10 degree disturbing only in one axis. yes
	// 0.9913    0.0811   -0.0597    0.0843  :10 degree disturbing only in each axis.
	// 0.9986    0.0329   -0.0221    0.0351 : 5 degree in each axis,

	R=q.normalized().toRotationMatrix();
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
	xi.translation()=t;
	cout << "\n Show initial pose:\n" << xi.rotationMatrix() << "\n Show translation:\n" << xi.translation()<<endl;




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
////		inliers_filter.emplace(210,292); //yes
////		inliers_filter.emplace(209,293); //yes
////		inliers_filter.emplace(208,294); //yes
////		inliers_filter.emplace(209,295); //yes
////		inliers_filter.emplace(208,296); //yes
////		inliers_filter.emplace(206,297); //yes
		inliers_filter.emplace(205,301); //yes








        int i=0;
		while ( i < 2){
			PhotometricBA(IRef, I, options, Klvl, xi, DRef,deltaMap);
			updateDelta(xi,Klvl,image_ref_baseColor,DRef,image_ref_metallic ,image_ref_roughness,light_source,deltaMap);




//			deltaMap.convertTo(deltaMap, CV_32FC1, 200.0);
//			imshow("deltamap", deltaMap);
//			waitKey(0);

			for(int x = 0; x < deltaMap.rows; ++x)
			{
				for(int y = 0; y < deltaMap.cols; ++y)
				{
					if(inliers_filter.count(x)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
					if(inliers_filter[x]!=y ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//					if (deltaMap.at<float>(x,y)==-nan){cout << "coord"<< }
					cout<<"show delta:"<<deltaMap.at<float>(x,y)<<endl;
					int tst=1;


				}
			}
//			Mat mask = cv::Mat(deltaMap != deltaMap);
//			deltaMap.setTo(1.0, mask);
			double  max, min;
			cv::minMaxIdx(deltaMap, &min, &max);
			cout<<"show max and min"<< max <<","<<min<<endl;


//			Mat result;
//			grayImage_ref.copyTo(result,mask);
//			imshow("show masked image", result);
//			waitKey(0);





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

