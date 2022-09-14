
// read image
// select pixel
// optimization
// pyramid improvement

//local header files
#include <reprojection.h>


//#include <algorithm>
//#include <atomic>
//#include <chrono>
#include <iostream>
#include <vector>
//#include <sstream>
#include <thread>
#include <cmath>
//#include <visnav/common_types.h>
#include <sophus/se3.hpp>
//#include <tbb/concurrent_unordered_map.h>
#include <unordered_map>
#include <Eigen/Core>
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
//#include <thread>


//using namespace visnav;
using namespace cv;
using namespace std;
using namespace DSONL;

bool checkImageBoundaries(Eigen::Vector2f &pixel, int width, int height);

bool removeNegativeValue(Mat& src, Mat& dst){
	dst = cv::max(src, 0);
	return true;
}

bool project(int uj, int vj, float iDepth, const Eigen::Matrix<float, 3, 3> &KRKinv, const Eigen::Matrix<float, 3, 1> &Kt,
        Eigen::Vector2f &pt2d);

void downscale(Mat &image, Mat &depth, Eigen::Matrix3d &K, int &level, Mat &image_d, Mat &depth_d, Eigen::Matrix3d &K_d);

float bilinearInterpolation(const Mat &image, const float &x, const float &y);

void calclErr(Mat &IRef, Mat &DRef, Mat &Image, Sophus::SE3f xi, Eigen::Matrix3f &K, Eigen::VectorXf &residuals);

void printAll(const double *arr, int n) {
	cout << "show value of n:" << n << endl;
	for (int i = 0; i < n / 10; i++) {
		cout << arr[i];
		cout << ((i + 1) % 20 ? ' ' : '\n');
	}
}

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


//struct GetPixelGrayValue {
//
//	GetPixelGrayValue(const double pixel_gray_val_in[1],
//	                  const Eigen::Vector2d &pixelCoor,
//	                  const Eigen::Matrix3d & K,
//	                  const int rows,
//	                  const int cols,
//	                  const std::vector<double> &vec_pixel_gray_values,
//	                  const std::vector<double> &img_ref_depth_values,
//	                  const std::vector<double> &img_ref_vec_values
//					  ) {
//		pixel_gray_val_in_[0] = pixel_gray_val_in[0];
//		rows_ = rows;
//		cols_ = cols;
//		pixelCoor_ = pixelCoor;
//		K_ = K;
////		interp_depth = interpolated_depth;
//
//		grid2d_depth.reset(new ceres::Grid2D<double>(&img_ref_depth_values[0],0, rows_, 0, cols_));
//		interp_depth.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double>>(*grid2d_depth));
//
//
//		grid2d_img_ref.reset(new ceres::Grid2D<double>(&img_ref_vec_values[0],0, rows_, 0, cols_));
//		interp_img_ref.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double>>(*grid2d_img_ref));
//
//
//		grid2d.reset(new ceres::Grid2D<double>(
//				&vec_pixel_gray_values[0], 0, rows_, 0, cols_));
//		get_pixel_gray_val.reset(
//				new ceres::BiCubicInterpolator<ceres::Grid2D<double> >(*grid2d));
//	}
//
//	template<typename T>
//	bool operator()(
//			const T* const sT,
////			const T* const  sd, //T const *const sd,
//			T *residual) const {
//
//		Eigen::Map<Sophus::SE3<T> const> const Tran(sT);
//		// project and search for optimization variable depth
//		// calculate transformed pixel coordinates
//		double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2);
//
//		Eigen::Matrix<double,3,1> p_3d_no_d;
////		Eigen::Vector3d p_3d_no_d = K_.inverse() * Eigen::Matrix<double, 3, 1>(pixelCoor_(1), pixelCoor_(2), 1.0);
//		p_3d_no_d<< (pixelCoor_(0)-cx)/fx, (pixelCoor_(1)-cy)/fy,1.0;
//
//
////		Eigen::Map<Eigen::Matrix<T, 1, 1> const> const ?????????????????????????????? do I need to map p_3d_no_d
//
////		Eigen::Vector3d p_3d_wod, p_3d_new_proj; //((u-cx)/fx, (v-cy)/fy,1.0);
////		p_3d_wod = K_.inverse() * Eigen::Matrix<double, 3, 1>(u, v, 1.0);
//        T d, u_, v_, intensity_image_ref;
//		u_=(T)pixelCoor_(1);
//		v_=(T)pixelCoor_(0);
//		interp_depth->Evaluate(u_,v_, &d);
//
//		interp_img_ref->Evaluate(u_,v_, &intensity_image_ref);
////		Eigen::Map<Eigen::Matrix<T, 1, 1> const> const d(sd);
////		Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(sResiduals);
//        Eigen::Matrix<T, 3,1> p_w=d*p_3d_no_d;
//		Eigen::Matrix<T, 3, 1> p1 = Tran * p_w ;
//		Eigen::Matrix<T, 3, 1> pt = K_ * p1;
//
//		T x = (pt[0] / pt[2]); // col id
//		T y = (pt[1] / pt[2]);// row id
//
////	    cout<<"\n Show current col id and row id:("<<x<<","<<y<<")"<<endl;
//		T pixel_gray_val_out;
//		get_pixel_gray_val->Evaluate(y, x, &pixel_gray_val_out); //
////	    cout<<"\n Show current pixel_gray_val_out:"<< pixel_gray_val_out<<endl;
//	residual[0] = intensity_image_ref - pixel_gray_val_out;
//
//
//		return true;
//	}
//
//
//	double pixel_gray_val_in_[1];
//	int rows_, cols_;
//	Eigen::Vector2d pixelCoor_;
//	Eigen::Matrix3d K_;
//	std::unique_ptr<ceres::Grid2D<double> > grid2d;
//	std::unique_ptr<ceres::Grid2D<double> > grid2d_depth;
//	std::unique_ptr<ceres::Grid2D<double> > grid2d_img_ref;
//	std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double> > > get_pixel_gray_val;
//	std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>> >interp_depth;
//	std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>> >interp_img_ref;
//};



void PhotometricBA(Mat &image, Mat &image_right, const PhotometricBAOptions &options, const Eigen::Matrix3d &K,
                   Sophus::SE3d &pose, Mat& depth_map);


int main(int argc, char **argv) {

	// loaded images
	string image_ref_path = "../data/rgb/1305031102.175304.png"; // data/rgb/1305031102.175304.png, data_test/rgb/1305031453.359684.png
	string image_target_path = "../data/rgb/1305031102.275326.png";  // matlab 1305031102.175304
	string depth_ref_path = "../data/depth/1305031102.160407.png";  //   matlab      1305031102.262886
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

	depth_ref.convertTo(depth_ref, CV_64FC1);
	depth_ref = depth_ref / 5000.0; // matlab 5000
//	 double min, max;
//	 cv::minMaxIdx(depth_ref, &min, &max);
//	 cout<<"show the depth_ref value range"<<"min:"<<min<<"max:"<<max<<endl;
//   depth_target.convertTo(depth_target, CV_64F);
//   depth_target = depth_target / 5000.0;

//
//	// left image
//	Mat grayImage_left,grayImage_l ;
//	Mat image_left = imread(image_l, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
//	cvtColor(image_left, grayImage_left, COLOR_RGB2GRAY);
//	grayImage_left.convertTo(grayImage_l, CV_64FC1, 1.0 / 255.0);
//
//	// right image
//	Mat grayImage_right,grayImage_r ;
//	Mat image_right = imread(image_r, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
//	cvtColor(image_right, grayImage_right, COLOR_RGB2GRAY);
//	grayImage_right.convertTo(grayImage_r, CV_64FC1, 1.0 / 255.0);
//
//	// left image depth
//	double depth_scale = 1000.0;
//	Mat depth_left = imread(depth_l, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
//	depth_left.convertTo(depth_left, CV_64FC1);
//	depth_left = depth_left / depth_scale;

//	imshow("Image_left", grayImage_ref);
//	imshow("Image_right", grayImage_target);
//	imshow("Image_left depth", depth_left);
//	waitKey(0);
//
//	double min, max;
//	 cv::minMaxIdx(grayImage_ref, &min, &max);
//	 cout<<"show the depth_ref value range"<<"min:"<<min<<"max:"<<max<<endl;





	// 相机内参
	Eigen::Matrix3d K;
	K << 517.3, 0, 318.6,
			0, 516.5, 255.3,
			0, 0, 1;

	double cx = 325.5;
	double cy = 253.5;
	double fx = 518.0;
	double fy = 519.0;

//	K<<fx,0,cx,0,fy,cy,0,0,1.0;


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
		PhotometricBA(IRef, I, options, Klvl, xi, DRef);
//		cout<<"optimized test value: "<<DRef.at<double>(363,376)<<endl;
		cout << "\n Show optimized pose:\n" << xi.rotationMatrix() << "\n Show translation:\n" << xi.translation()
		     << endl;
		Eigen::Quaterniond q_opt( xi.rotationMatrix());
		cout<<"\n Show the optimized rotation as quaternion:"<<q_opt.w()<<","<<q_opt.x()<<","<<q_opt.y()<<","<<q_opt.z()<< endl;

	}

	return 0;
}

void
downscale(Mat &image, Mat &depth, Eigen::Matrix3d &K, int &level, Mat &image_d, Mat &depth_d, Eigen::Matrix3d &K_d) {
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

bool checkImageBoundaries(Eigen::Vector2f &pixel, int width, int height) {
	return (pixel[0] > 1.1 && pixel[0] < width - 2.1 && pixel[1] > 1.1 && pixel[1] < height - 2.1);
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

void calclErr(Mat &IRef, Mat &DRef, Mat &Image, Sophus::SE3f xi, Eigen::Matrix3f &K, Eigen::VectorXf &residuals) {

	const Eigen::Matrix3f KRKinv = K * xi.rotationMatrix() * K.inverse();
	const Eigen::Vector3f Kt = K * xi.translation();

	for (int u = 0; u < IRef.rows; u++) {
		for (int v = 0; v < IRef.cols; v++) {

			Eigen::Vector2f pt2d;
			float depth = DRef.at<float>(u, v);
			float newIntensity;
			const Eigen::Matrix<float, 3, 1> pt = KRKinv * Eigen::Matrix<float, 3, 1>(u, v, 1) * depth + Kt;
			if (pt[2] > 0 && depth > 0) {
				pt2d[0] = pt[0] / pt[2];
				pt2d[1] = pt[1] / pt[2];
			}

			// obtain bilinear interpolated intensity values
			// newIntensity = bilinearInterpolation(Image, pt2d[0], pt2d[1]);

			if (newIntensity > 0) {
				residuals[u * IRef.rows + v] = IRef.at<float>(u, v) - bilinearInterpolation(Image, pt2d[0], pt2d[1]);
			}
		}
	}
}

void PhotometricBA(Mat &image, Mat &image_right, const PhotometricBAOptions &options, const Eigen::Matrix3d &K,
                   Sophus::SE3d &pose, Mat &img_ref_depth) {
	ceres::Problem problem;
	// Setup optimization problem
	// convert rigth image into double type vector
	cv::Mat flat = image_right.reshape(1, image_right.total() * image_right.channels());
	std::vector<double> img_gray_values = image_right.isContinuous() ? flat : flat.clone();

	cv::Mat flat_depth_map = img_ref_depth.reshape(1, img_ref_depth.total() * img_ref_depth.channels());
	std::vector<double> img_ref_depth_values=img_ref_depth.isContinuous() ? flat_depth_map : flat_depth_map.clone();

	cv::Mat flat_ref = image.reshape(1, image.total() * image.channels());
	std::vector<double> image_ref_vec = image.isContinuous() ? flat_ref : flat_ref.clone();


	problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters, new Sophus::test::LocalParameterizationSE3);

	std::unordered_map<int, int> inliers_filter;
	//new image
	inliers_filter.emplace(213,248); //yes
	inliers_filter.emplace(280,411); //yes
	inliers_filter.emplace(112,304); //yes
	inliers_filter.emplace(121,231); //yes
	inliers_filter.emplace(312,180); //yes


//	inliers_filter.emplace(159,294); //yes
//

//	inliers_filter.emplace(256,67); //yes
//	inliers_filter.emplace(255,69);//yes
//	inliers_filter.emplace(252,76);//yes
//
//	//telephone_surface
//	inliers_filter.emplace(254,191);//0.012 ,yes
//	inliers_filter.emplace(243,190);//-0.0005.yes
//	inliers_filter.emplace(241,191);//0.008 yes
//
//	inliers_filter.emplace(449,383);//yes
//	inliers_filter.emplace(319,331);//yes
//	inliers_filter.emplace(288,327);//yes
//	inliers_filter.emplace(432,86);//yes
//	inliers_filter.emplace(459,80);//yes
//	inliers_filter.emplace(293,535);//yes
//
//    inliers_filter.emplace(310,540);//yes
//	inliers_filter.emplace(308,555);//yes
//	inliers_filter.emplace(307,548);//yes
//	inliers_filter.emplace(324,376);//yes
//	inliers_filter.emplace(324,231);//yes
//	inliers_filter.emplace(121,93);//yes
//	inliers_filter.emplace(131,104);//yes


	double depth_para,intensity_l ,gray_values[1]{};
	double *transformation = pose.data();
	// use pixels and depth to optimize pose and depth itself
	for (int u = 0; u< image.rows; u++) // colId, cols: 0 to 640
	{
		for (int v = 0; v < image.cols; v++) // rowId,  rows: 0 to 480
		{
			// use the inlier filter
//		   if(inliers_filter.count(u)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//		   if(inliers_filter[u]!=v ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//		   cout<<" \n show the coordinates:"<<u<<","<<v<<"---> value:"<<image.at<double>(u,v)<<endl; // checked already// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
			if (img_ref_depth.at<double>(u,v) < 1e-3 ) { continue; } //&& p_3d_new_proj(2)< 1e-4
			gray_values[0] =  image.at<double>(u, v);
			Eigen::Vector2d pixelCoord((double)v,(double)u);//  u is the row id , v is col id

					problem.AddResidualBlock(
							new ceres::AutoDiffCostFunction<GetPixelGrayValue, 1, Sophus::SE3d::num_parameters>(
									new GetPixelGrayValue(gray_values,
									                      pixelCoord,
									                      K,
									                      image.rows,
									                      image.cols,
									                      img_gray_values,
									                      img_ref_depth_values,
									                      image_ref_vec
									)
							),
							new ceres::HuberLoss(4.0/255.0), //   new ceres::HuberLoss(4.0/255.0),      // matlab (4.0/255.0)
							transformation
//							&testDirect
							);

		}
	}
	// Solve
	std::cout << "\n Solving ceres directBA ... " << endl;
	ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = 10000;
//	ceres_options.gradient_check_numeric_derivative_relative_step_size=1e-4;

	ceres_options.linear_solver_type =ceres::SPARSE_SCHUR; // ceres::SPARSE_SCHUR;  DENSE_NORMAL_CHOLESKY;
	ceres_options.num_threads = std::thread::hardware_concurrency();
	ceres_options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	Solve(ceres_options, &problem, &summary);
	switch (options.verbosity_level) {
		// 0: silent
		case 1:
			std::cout << summary.BriefReport() << std::endl;
			break;
		case 2:
			std::cout << summary.FullReport() << std::endl;
			break;
	}


}