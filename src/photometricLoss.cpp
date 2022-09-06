
// read image
// select pixel
// optimization
// pyramid improvement

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

bool checkImageBoundaries(Eigen::Vector2f &pixel, int width, int height);

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


//		Eigen::Map<Eigen::Matrix<T, 1, 1> const> const ?????????????????????????????? do I need to map p_3d_no_d

//		Eigen::Vector3d p_3d_wod, p_3d_new_proj; //((u-cx)/fx, (v-cy)/fy,1.0);
//		p_3d_wod = K_.inverse() * Eigen::Matrix<double, 3, 1>(u, v, 1.0);
        T d, u_, v_, intensity_image_ref;
		u_=(T)pixelCoor_(0);
		v_=(T)pixelCoor_(1);
		interp_depth->Evaluate(u_,v_, &d);

		interp_img_ref->Evaluate(u_,v_, &intensity_image_ref);
//		Eigen::Map<Eigen::Matrix<T, 1, 1> const> const d(sd);
//		Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(sResiduals);

		Eigen::Matrix<T, 3, 1> p1 = Tran * p_3d_no_d *d;
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


// struct PhotometricCostFunctor{
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//     PhotometricCostFunctor( const float& intensity_l, const Eigen::Vector3f& p_3f, const Eigen::Matrix3f& K, const Mat& image_left): intensity_l(intensity_l), p_3f(p_3f),K(K), image_left(image_left){}


//     template <class T>
//     bool operator()(T const* const sT, T const* const sd, T* sResiduals) const {

//     // map inputs
//     Eigen::Map<Sophus::SE3<T> const> const Tran(sT);
//     Eigen::Map<Eigen::Matrix<T, 1, 1> const> const d(sd);
//     Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(sResiduals);

//     // float* p_= const_cast<float*>(p.data());
//     // Eigen::Map<Eigen::Matrix<T, 3, 1> const> const p_3f(p_);

//     // p=p*d;
//     // float* p_= const_cast<float*>(p.data());
//     // //float* p_w = p.data();
//     // Eigen::Map<Eigen::Matrix<T, 3, 1> const> const p_3f(* p_w );
//     // Eigen::Map<Eigen::Matrix<T, 3, 1> const> const p_3f_w(sp_3f_w);
//     Eigen::Matrix<float, 3, 1> p1= Tran*p_3f*d;
//     Eigen::Matrix<float, 3, 1> pt= K*p1;
//     const float x= (pt[0] /pt[2]);
//     const float y=  (pt[0] /pt[2]) ;
//     float intensity_right= bilinearInterpolation(image_left,x ,y);
//     Eigen::Matrix<T, 1, 1>intensity_l= Eigen::Matrix<T, 1, 1>(intensity_l);
//     Eigen::Matrix<T, 1, 1>intensity_r=  Eigen::Matrix<T, 1, 1>(intensity_right);
//     residuals= T(intensity_l-intensity_r);
//     return true;
//     }


//     Eigen::Vector3f p_3f;
//     Eigen::Matrix3f K;
//     Mat image_left;
//     float intensity_l;
// };


void PhotometricBA(Mat &image, Mat &image_right, const PhotometricBAOptions &options, const Eigen::Matrix3d &K,
                   Sophus::SE3d &pose, Mat& depth_map);


int main(int argc, char **argv) {

	// loaded images
	string image_ref_path = "../data/rgb/1305031102.275326.png"; // matlab 1305031102.275326


	string image_target_path = "../data/rgb/1305031102.175304.png";  // matlab 1305031102.175304
	string depth_ref_path = "../data/depth/1305031102.262886.png";  //   matlab      1305031102.262886
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

	// float min;
	// float max;
	// cv::minMaxIdx(grayImage_ref, &min, &max);
	// cout << "show the grayImage value range"
	//      << "min:" << min << "max:" << max << endl;

	Mat depth_ref = imread(depth_ref_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	Mat depth_target = imread(depth_target_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);


	depth_ref.convertTo(depth_ref, CV_64FC1);
	depth_ref = depth_ref / 5000.0; // matlab 5000
//	double min, max;
//	 cv::minMaxIdx(depth_ref, &min, &max);
//	 cout<<"show the depth_ref value range"<<"min:"<<min<<"max:"<<max<<endl;
//    depth_target.convertTo(depth_target, CV_64F);
//    depth_target = depth_target / 5000.0;

	// 相机内参
	double cx = 325.5;
	double cy = 253.5;
	double fx = 518.0;
	double fy = 519.0;
	double depth_scale = 1000.0;
	Eigen::Matrix3d K;
//	K<<fx,0.0,cx,0.0,fy,cy,0.0,0.0,1.0;


	K << 517.3, 0, 318.6,
			0, 516.5, 255.3,
			0, 0, 1;

	Sophus::SE3d xi;
	Sophus::Vector6d groundTruth_pose;
	groundTruth_pose << -0.0032, 0.0065, 0.0354, -0.0284, -0.0172, -0.0011;
	Sophus::SE3d groundTruth_pose_xi = Sophus::SE3d::exp(groundTruth_pose);

	cout << "\n Show initial pose:\n" << groundTruth_pose_xi.rotationMatrix() << "\n Show translation:\n" << groundTruth_pose_xi.translation()<<endl;

//	xi = groundTruth_pose_xi; // initialize the pose value
	Sophus::Vector6d log_xi_init = xi.log();
	cout << "\n Show initial Lie algebra of pose:\n" << log_xi_init << endl;

	int lvl_target, lvl_ref;
	for (int lvl = 1; lvl >= 1; lvl--)// ???????????????????
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



//
		double  test_depth=DRef.at<double>(363,376);
		cout<<"test value: "<<test_depth<<endl;
		PhotometricBAOptions options;
		PhotometricBA(IRef, I, options, Klvl, xi, DRef);
		cout<<"optimized test value: "<<DRef.at<double>(363,376)<<endl;
		cout << "\n Show optimized pose:\n" << xi.rotationMatrix() << "\n Show translation:\n" << xi.translation()
		     << endl;
		Sophus::Vector6d log_xi = xi.log();
		cout << "\n Show Lie algebra of optimized pose:\n" << log_xi.transpose() << endl;


	}

	return 0;
}

void
downscale(Mat &image, Mat &depth, Eigen::Matrix3d &K, int &level, Mat &image_d, Mat &depth_d, Eigen::Matrix3d &K_d) {
//	imshow("depth", depth);
//	waitKey(0);
	if (level <= 1) {
		image_d = image;
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




//	ceres::Grid2D<double> depth_grid(&img_ref_depth_values[0], 0, image.rows, 0, image.cols);
//	ceres::BiCubicInterpolator<ceres::Grid2D<double>> interp_depth(depth_grid);

	problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters, new Sophus::test::LocalParameterizationSE3);

	std::unordered_map<int, int> inliers_filter;
	//position: keyboard_key_esc:
	inliers_filter.emplace(363,376);
	inliers_filter.emplace(364,379);
	inliers_filter.emplace(361,378);
	//position: USB_plug:
	inliers_filter.emplace(242,442);
	inliers_filter.emplace(242,440);
	inliers_filter.emplace(239,441);
	//Hole punch:
//	inliers_filter.emplace(281,258);
//	inliers_filter.emplace(283,256);
//	inliers_filter.emplace(277,259);










//	for (double v = 0; v < image.cols; v++) //
//	{
//		for (double u = 0; u < image.rows; u++)
//		{
//			// use the inlier filter
//			if(inliers_filter.count(u)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//		if(inliers_filter[u]!=v ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//			double depth_para;
//			interp_depth.Evaluate(u, v, &depth_para);
//			if (depth_para < 1e-3 ) { continue; }
//			cout << "\n Check depth value of ceres Grid2D:\n" << depth_para<< endl;
//			problem.AddParameterBlock(&depth_para, 1);
//		}
//	}
//
//	for (int v = 0; v < image.cols; v++) // ???????????????????????????????????????
//	{
//		if (v != 378) { continue; }
//		for (int u = 0; u < image.rows; u++)
//		{
//			if (u != 361) { continue; }//
//			double intensity_value[1]{};
//			intensity_value[0] = image.at<double>(u,v);
//			problem.AddParameterBlock(intensity_value, 1);
//			problem.SetParameterBlockConstant(intensity_value);
//		}
//	}


	double depth_para,intensity_l ,gray_values[1]{};
	double *transformation = pose.data();

//	double  depth_para_test=1.88888;
//	double  testDirect= depth_para_test;

	// use pixels and depth to optimize pose and depth itself
	for (int v = 0; v < image.cols; v++) // colId, cols: 0 to 640
	{
//		if (v != 378) { continue; }// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		for (int u = 0; u < image.rows; u++) // rowId,  rows: 0 to 480
		{



//			// use the inlier filter
//		   if(inliers_filter.count(u)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//		   if(inliers_filter[u]!=v ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//		   cout<<" \n show the coordinates:"<<u<<","<<v<<"---> value:"<<image.at<double>(u,v)<<endl; // checked already// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~

			if (img_ref_depth.at<double>(u,v) < 1e-3 ) { continue; } //&& p_3d_new_proj(2)< 1e-4
			gray_values[0] =  image.at<double>(u, v);
			Eigen::Vector2d pixelCoord((double)u,(double)v);//    !!!!!!!!!!!!!!!!   u is the row id , v is col id
//			interp_depth.Evaluate(u,v, &testDirect); // depth_para---->depth_para_test





//			depth_para[0]*=1000;  test if we can change the value of grid2d
//			double depth_para_modified[1]{};
//			interp_depth.Evaluate(u,v, &depth_para_modified[0]);
//			cout<<"show modified depth value at ( 361, 378):"<<depth_para_modified[0]<<endl;
//	        cout<<"\n Show current intensity_l:"<< intensity_l<<endl;
//			double p_depth_w = depth_map.at<double>(u, v);
//			Eigen::Vector3d p_3d_wod, p_3d_new_proj;
//			p_3d_wod = K.inverse() * Eigen::Matrix<double, 3, 1>(u, v, 1.0);
//			p_3d_new_proj=(K*(pose*p_3d_wod*p_depth_w));
//			double u_p= p_3d_new_proj(0)/p_3d_new_proj(2);
//			double v_p= p_3d_new_proj(1)/p_3d_new_proj(2);
//			depth_para_test[0]=1.414;
//			double * testDepth= const_cast<double *>(dep.data());
//			cout<<"show fake data:"<<depth_para_test<<": data"<<depth_para_test[0]<<endl;
//			double xnwe[] = {1.414};
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
							new ceres::HuberLoss(0.1*4.0/255.0), // matlab (4.0/255.0)
							transformation
//							&testDirect
							);

		}
	}
	// Solve
	std::cout << "\n Solving ceres directBA ... " << endl;
	ceres::Solver::Options ceres_options;
	ceres_options.sparse_linear_algebra_library_type=ceres::SUITE_SPARSE;
    ceres_options.max_num_iterations = 50;
	ceres_options.use_explicit_schur_complement= true;
	ceres_options.linear_solver_type =ceres::SPARSE_SCHUR; // ceres::SPARSE_SCHUR;  DENSE_NORMAL_CHOLESKY;
//	ceres_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
//	ceres_options.preconditioner_type = ceres::SCHUR_JACOBI;
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