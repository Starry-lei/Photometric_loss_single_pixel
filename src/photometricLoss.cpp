
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
//#include "opencv2/features2d.hpp"


#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>
#include <visnav/local_parameterization_se3.hpp>

using namespace cv;
using namespace std;
using namespace DSONL;

const double DEG_TO_ARC = 0.0174532925199433;

int main(int argc, char **argv) {

	Eigen::Matrix<double,3,1> light_source(-32.4, -27 ,31.8);
	Eigen::Matrix<double,3,1> Camera1(3.8, -16.5 ,26.1);
	Eigen::Matrix<double,3,1> Camera2(0.3 ,-16.9, 27.7);



	double roll_arc_c1 = -178.917* DEG_TO_ARC;      // 绕X轴
	double pitch_arc_c1 = 0* DEG_TO_ARC;     // 绕Y轴
	double yaw_arc_c1 = -3.793* DEG_TO_ARC;     // 绕Z轴

	double roll_arc_c2 = -179.988* DEG_TO_ARC;      // 绕X轴
	double pitch_arc_c2 = 0.263* DEG_TO_ARC;     // 绕Y轴
	double yaw_arc_c2 = -3.972* DEG_TO_ARC;     // 绕Z轴
	Eigen::Vector3d euler_angle1(roll_arc_c1, pitch_arc_c1, yaw_arc_c1);
	Eigen::Vector3d euler_angle2(roll_arc_c2, pitch_arc_c2, yaw_arc_c2);
	Eigen::Matrix3d rotation_matrix1, rotation_matrix2;
	rotation_matrix1 = Eigen::AngleAxisd(euler_angle1[2], Eigen::Vector3d::UnitZ()) *
	                   Eigen::AngleAxisd(euler_angle1[1], Eigen::Vector3d::UnitY()) *
	                   Eigen::AngleAxisd(euler_angle1[0], Eigen::Vector3d::UnitX());
	rotation_matrix2 = Eigen::AngleAxisd(euler_angle2[2], Eigen::Vector3d::UnitZ()) *
	                   Eigen::AngleAxisd(euler_angle2[1], Eigen::Vector3d::UnitY()) *
	                   Eigen::AngleAxisd(euler_angle2[0], Eigen::Vector3d::UnitX());





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






	Eigen::Vector3d light_source_c1;
	light_source_c1= light_source-Camera1; //TODO: remains to be subtracted by p_3d and normalised;


//	// loaded images
//	string image_ref_path = "../data/rgb/1305031102.175304.png"; // data/rgb/1305031102.175304.png, data_test/rgb/1305031453.359684.png
//	string image_target_path = "../data/rgb/1305031102.275326.png";  // matlab 1305031102.175304
//	string depth_ref_path = "../data/depth/1305031102.160407.png";  //   matlab      1305031102.262886
	string image_ref_path = "../data/rgb/viewpoint1_rgb.png";
	string image_target_path = "../data/rgb/viewpoint2_rgb.png";
	string depth_ref_path = "../data/depth/viewpoint2.exr";

	// read metallic adn roughness data// read metallic adn roughness data
	string image_ref_MR_path = "../data/rgb/viewpoint1_mr.png"; // store value in rgb channels,  channel b: metallic, channel green: roughness
	string image_target_MR_path = "../data/rgb/viewpoint2_mr.png";
	//  create a metallic and roughness table, (map R G values into [0 ,1] for two images)
	// create a metallic and roughness table for reference image
	Mat image_ref_MR= imread(image_ref_MR_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    Mat ref_mr_table[3];
	split(image_ref_MR,ref_mr_table);// 0: red, 1: green, 2: blue
	Mat image_ref_metallic= ref_mr_table[2];
	Mat image_ref_roughness= ref_mr_table[1];
	image_ref_metallic.convertTo(image_ref_metallic, CV_64FC1,1.0 / 255.0);
	image_ref_roughness.convertTo(image_ref_roughness, CV_64FC1,1.0 / 255.0);
    //  create a metallic and roughness table for target image
	Mat image_target_MR= imread(image_target_MR_path,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	Mat taget_mr_table[3];
	split(image_target_MR,taget_mr_table);// 0: red, 1: green, 2: blue
	Mat image_target_metallic= taget_mr_table[2];
	Mat image_target_roughness= taget_mr_table[1];
	image_target_metallic.convertTo(image_target_metallic, CV_64FC1,1.0 / 255.0);
	image_target_roughness.convertTo(image_target_roughness, CV_64FC1,1.0 / 255.0);






	// read base color data TODO: check if we need to map the value of baseColor
	string image_ref_baseColor = "../data/rgb/viewpoint1_texture.png";
	string image_target_baseColor = "../data/rgb/viewpoint2_texture.png";









	//	string image_ref_path = "../data_test/rgb/1305031117.243277.png"; //  , data_test/rgb/1305031453.359684                data_test/rgb/1305031453.359684.png
	//	string image_target_path = "../data_test/rgb/1305031117.843291.png";  // matlab 1305031102.175304
	//	string depth_ref_path = "../data_test/depth/1305031117.241340.png";  //   matlab      1305031102.262886

	string depth_target_path = "../data/depth/1305031102.160407.png";
	Mat grayImage_target, grayImage_ref;
	// read target image
	Mat image_target = imread(image_target_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	// color space conversion
	cvtColor(image_target, grayImage_target, COLOR_RGB2GRAY);
	// precision improvement
	grayImage_target.convertTo(grayImage_target, CV_64FC1, 1.0 / 255.0);
	// read ref image
	Mat image_ref = imread(image_ref_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	// color space conversion
	cvtColor(image_ref, grayImage_ref, COLOR_RGB2GRAY);
	// precision improvement
	grayImage_ref.convertTo(grayImage_ref, CV_64FC1, 1.0 / 255.0);
	Mat depth_ref = imread(depth_ref_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	Mat depth_target = imread(depth_target_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	Mat channel[3];
	split(depth_ref,channel);
	depth_ref=channel[0];
	depth_ref= depth_ref *(60.0-0.01) + 0.01; // /255.0
	depth_ref.convertTo(depth_ref, CV_64FC1);

//   double min, max;
//   cv::minMaxIdx(depth_ref, &min, &max);
//   cout<<"show the depth_ref value range"<<"min:"<<min<<"max:"<<max<<endl;

//   depth_target.convertTo(depth_target, CV_64F);
//   depth_target = depth_target / 5000.0;


//	double min, max;
//	 cv::minMaxIdx(grayImage_ref, &min, &max);
//	 cout<<"show the depth_ref value range"<<"min:"<<min<<"max:"<<max<<endl;

	// 相机内参
	Eigen::Matrix3d K;
//	K << 517.3, 0, 318.6,
//			0, 516.5, 255.3,
//			0, 0, 1;
//	double cx = 325.5;
//	double cy = 253.5;
//	double fx = 518.0;
//	double fy = 519.0;

//	K<<fx,0,cx,0,fy,cy,0,0,1.0;

 K<< 800.0, 0, 0,
     0, 800.0, 0,
	 0,   0,  1;
// TODO: test the current pose and compare it with the one using function
	Mat normalMap=getNormals(K,depth_ref);

	Sophus::SE3d xi;
	Eigen::Matrix<double, 3,3> R;
	Eigen::Matrix<double, 3,1> t;

//	R<< 0.9990,  0.0210 ,   0.0395,
//	   -0.0219 ,   0.9995 ,  0.0237,
//	    -0.0389,   -0.0246,  0.9989;


//	R = 0.7202, 0.2168, 0.6591,
//	        -0.4602,  0.8601,  0.2200,
//			-0.5192,  -0.4617, 0.7192;
// GT
	Eigen::Quaterniond q(0.9998,    0.0174 ,   0.0076 ,  -0.0003);
	// disturbing rotation
//	Eigen::Quaterniond q( 0.9998,    0.0174 ,   0.0076 ,  -0.0003);
    //	0.9949 ,  -0.0103  ,  0.0206 ,  -0.0978 : 10 degree disturbing only in one axis. yes
	// 0.9913    0.0811   -0.0597    0.0843  :10 degree disturbing only in each axis.
	// 0.9986    0.0329   -0.0221    0.0351 : 5 degree in each axis,

	R=q.normalized().toRotationMatrix();
// GT
	t<<   0.0028,
	      -0.0053,
	      -0.0362;
	// disturbing translation
//	t<<  0.0223,
//	     0.1023,
//		 -0.0246;

//	xi.setRotationMatrix(R);
//	xi.translation()=t;
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

		PhotometricBAOptions options;
		PhotometricBA(IRef, I, options, Klvl, xi, DRef, light_source_c1);
//		cout<<"optimized test value: "<<DRef.at<double>(363,376)<<endl;
		cout << "\n Show optimized pose:\n" << xi.rotationMatrix() << "\n Show translation:\n" << xi.translation()
		     << endl;
		Eigen::Quaterniond q_opt( xi.rotationMatrix());
		cout<<"\n Show the optimized rotation as quaternion:"<<q_opt.w()<<","<<q_opt.x()<<","<<q_opt.y()<<","<<q_opt.z()<< endl;

	}

	return 0;
}




