//
// Created by cheng on 12.09.22.
//
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unordered_map>
using namespace cv;
using namespace std;

int main() {

  // loaded images
  string image_ref_path =
      "../data/rgb/1305031102.275326.png"; // matlab 1305031102.275326

  string image_target_path =
      "../data/rgb/1305031102.175304.png"; // matlab 1305031102.175304
  string depth_ref_path =
      "../data/depth/1305031102.262886.png"; //   matlab      1305031102.262886
  string depth_target_path = "../data/depth/1305031102.160407.png";
  Mat grayImage_target, grayImage_ref;

  // read target image
  Mat image_target =
      imread(image_target_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
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

  Mat depth_ref =
      imread(depth_ref_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
  Mat depth_target = imread(depth_target_path,
                            CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

  depth_ref.convertTo(depth_ref, CV_64FC1);
  depth_ref = depth_ref / 5000.0; // matlab 5000
                                  //	double min, max;
                                  //	 cv::minMaxIdx(depth_ref, &min, &max);
  //	 cout<<"show the depth_ref value range"<<"min:"<<min<<"max:"<<max<<endl;
  // 相机内参
  Eigen::Matrix3d K;
  K << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1;
}