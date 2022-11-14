
// read image
// select pixel
// optimization
// pyramid improvement

//local header files
#include "reprojection.h"
#include "photometricBA.h"
#include "ultils.h"
#include "PCLOpt.h"
#include "dataLoader.h"
#include "pixelSelector.h"

//#include <algorithm>
//#include <atomic>
//#include <chrono>

#include <sophus/se3.hpp>
//#include <tbb/concurrent_unordered_map.h>
#include <unordered_map>
//#include <Eigen/Dense>
#include <Eigen/Core>
//#include <Eigen/Geometry>

//#include <Eigen/StdVector>
//#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
//#include "opencv2/features2d.hpp"



//#include <ceres/ceres.h>
//#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>
//#include <visnav/local_parameterization_se3.hpp>

using namespace cv;
using namespace std;
using namespace DSONL;


int main(int argc, char **argv){


	Eigen::Vector2i pixel_pos(213,295);
//	std::unordered_map<int, int> inliers_filter;
//	inliers_filter.emplace(173,333); //yes
//	inliers_filter.emplace(378,268); //yes

	// =======================================================data loader========================================
	Mat grayImage_target, grayImage_ref,depth_ref,depth_target,image_ref_baseColor,image_target_baseColor;
	dataLoader* dataLoader;
	dataLoader= new DSONL::dataLoader();
	dataLoader->Init();

	float image_ref_metallic = dataLoader->image_ref_metallic;
	float image_ref_roughness= dataLoader->image_ref_roughness;

	grayImage_ref=dataLoader->grayImage_ref;
	grayImage_target=dataLoader->grayImage_target;
	grayImage_ref.convertTo(grayImage_ref,CV_64FC1);
	grayImage_target.convertTo(grayImage_target, CV_64FC1);

	depth_ref=dataLoader->depth_map_ref;

	Mat depth_ref_GT= dataLoader->depth_map_ref;
	depth_target=dataLoader->depth_map_target;
	image_ref_baseColor= dataLoader->image_ref_baseColor;
	image_target_baseColor= dataLoader->image_target_baseColor;


	// show the depth image with noise
	double min_depth_val, max_depth_val;
	cv::minMaxLoc(depth_ref, &min_depth_val, &max_depth_val);
	cout<<"\n show original depth_ref min, max:\n"<<min_depth_val<<","<<max_depth_val<<endl;


	Eigen::Matrix3f K;
	K=dataLoader->camera_intrinsics;




													imshow("grayImage_ref",grayImage_ref);
													imshow("grayImage_target",grayImage_target);
													waitKey(0);















	// ----------------------------------------optimization variable: R, t--------------------------------------
	Sophus::SE3d xi, xi_GT;
//	Sophus::SO3d Rotation;
//	Eigen::Matrix<double, 3,1> Translation;

	Eigen::Matrix<double, 3,3> R;
	R=dataLoader->q_12.normalized().toRotationMatrix();
	xi_GT.setRotationMatrix(R);
	xi_GT.translation()=dataLoader->t12;

	// ----------------------------------------optimization variable: depth --------------------------------------
	cout << "\n Show GT rotation:\n" << xi_GT.rotationMatrix() << "\n Show GT translation:\n" << xi_GT.translation()<<endl;

// ------------------------------------------------------------------------------------------Movingleast algorithm---------------------------------------------------------------
	std::vector<Eigen::Vector3d> pts;
	cv::Mat normal_map(depth_ref.rows, depth_ref.cols, CV_64FC3);
	//MLS();
//	---------------------------------------------------------normal_map_GT---------------------------------------------------
	Mat normal_map_GT;
	normal_map_GT=dataLoader->normal_map_GT;
	for (int u = 0; u< depth_ref.rows; u++) // colId, cols: 0 to 480
	{
		for (int v = 0; v < depth_ref.cols; v++) // rowId,  rows: 0 to 640
		{

			Eigen::Vector3d normal_new( normal_map_GT.at<cv::Vec3f>(u,v)[2],  -normal_map_GT.at<cv::Vec3f>(u,v)[1], normal_map_GT.at<cv::Vec3f>(u,v)[0]);
			normal_new= dataLoader->R1.transpose()*normal_new;

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


//	--------------------------------------------------------------------Data perturbation--------------------------------------------------------------------
// Add noise to original depth image, depth_ref_NS
	Mat inv_depth_ref, depth_ref_gt;
	Mat depth_ref_NS;
	double roErr;
	Eigen::Matrix3d R_GT(xi_GT.rotationMatrix());
	Eigen::Matrix3d perturbedRotation=rotation_pertabation(0.0,0.0,0.0,R_GT,roErr); // degree

	double trErr;
	Eigen::Vector3d T_GT(xi_GT.translation());
	Eigen::Vector3d  perturbedTranslation=translation_pertabation(0.0, 0.0, 0.0, T_GT,trErr); // percentage

	double Mean=0.0,StdDev=0;
	float densities[] = {0.03,0.003, 0.05,0.15,0.5,1}; /// number of optimized depths,  current index is 1



	PhotometricBAOptions options;
	Mat newNormalMap=normal_map;
	double distanceThres=0.07;
	float upper=5;
	float buttom=0.2;
	float up_new=upper;
	float butt_new=buttom;
	Mat deltaMap(depth_ref.rows, depth_ref.cols, CV_32FC1, Scalar(1)); // storing delta
	int lvl_target, lvl_ref;

	double depth_upper_bound = 0.2;  // 0.5; 1
	double depth_lower_bound = 0.0001;  // 0.001

	options.optimize_depth = false;
	options.useFilterController= false; // control the number of optimized depth
	options.optimize_pose= true;
	options.use_huber= true;
	options.lambertianCase= false;
	options.usePixelSelector= false;
	dataLoader->options_.remove_outlier_manually= false;
	options.huber_parameter=  0.25*4.0/255.0;   /// 0.25*4/255 :   or 4/255

	// initialize the pose xi
	xi.setRotationMatrix(perturbedRotation);
	xi.translation()=perturbedTranslation;



	Sophus::SO3d Rotation(xi.rotationMatrix());
	Eigen::Matrix<double, 3,1> Translation(xi.translation());



	PixelSelector* pixelSelector=NULL;
	FrameHessian* newFrame_ref=NULL;
	FrameHessian* newFrame_tar=NULL;
	FrameHessian* depthMap_ref=NULL;
	float* color_ref=NULL;
	float* color_tar=NULL;
	float* depthMapArray_ref=NULL;
	float* statusMap=NULL;
	bool*  statusMapB=NULL;

	if(options.useFilterController){
		Mat grayImage_ref_CV8U;
		grayImage_ref.convertTo(grayImage_ref_CV8U,CV_8UC1,255.0);
		newFrame_ref= new FrameHessian();
		pixelSelector= new PixelSelector(wG[0],hG[0]);
		color_ref= new float[wG[0]*hG[0]];
		for (int row = 0; row < hG[0]; ++row) {
			uchar *pixel_ref=grayImage_ref_CV8U.ptr<uchar>(row);
			for (int col = 0; col < wG[0]; ++col) {
				color_ref[row*wG[0]+col]= (float) pixel_ref[col];
			}
		}
		newFrame_ref->makeImages(color_ref); // make image_ref pyramid
		statusMap= new float[wG[0]*hG[0]];


		int npts= pixelSelector->makeMaps(newFrame_ref, statusMap, densities[1]*wG[0]*hG[0], 1, false, 2);
		cerr<<"npts:"<<npts<<endl;


	}

//	AddGaussianNoise_Opencv(depth_ref,depth_ref_NS,Mean,StdDev);
	AddGaussianNoise_Opencv(depth_ref,depth_ref_NS,Mean,StdDev,statusMap);
//	depth_ref_NS.at<double>(319,296)*=(1.0+ 0.05) ;
//	depth_ref_NS.at<double>(319,296)*=(1.0+ 0.05) ;


	divide(Scalar(1), depth_ref, depth_ref_gt);
	divide(Scalar(1), depth_ref_NS, inv_depth_ref);
	Mat depth_ref_NS_before=inv_depth_ref.clone();


	double min_inv, max_inv;
	cv::minMaxLoc(inv_depth_ref, &min_inv, &max_inv);
	cout<<"\n show original inv_depth_ref min, max:\n"<<min_inv<<","<<max_inv<<endl;
	Scalar_<double> depth_Err=depthErr(depth_ref_gt, inv_depth_ref);
	double depth_Error=depth_Err.val[0];
	cout << "\n Show initial rotation:\n" << Rotation.matrix() << "\n Show initial translation:\n" <<Translation<<endl;
	cout<<"\nShow current rotation perturbation error :"<< roErr<< "\n Show current translation perturbation error : "<< trErr<<"\nShow current depth perturbation error :"<< depth_Error<<endl;



	if (options.usePixelSelector){
		// =========================================================Pixel Selector(left image)==============================================================
		// using float type in PixelSelector

		double min_gray,max_gray;
		Mat grayImage_ref_CV8U;
		Mat grayImage_tar_CV8U;
		grayImage_ref.convertTo(grayImage_ref_CV8U,CV_8UC1,255.0);
		grayImage_target.convertTo(grayImage_tar_CV8U,CV_8UC1,255.0);

		newFrame_ref= new FrameHessian();
		newFrame_tar= new FrameHessian();
		depthMap_ref= new FrameHessian();

		pixelSelector= new PixelSelector(wG[0],hG[0]);
		color_ref= new float[wG[0]*hG[0]];
		color_tar= new float[wG[0]*hG[0]];

		depthMapArray_ref= new float[wG[0]*hG[0]];
		inv_depth_ref.convertTo(inv_depth_ref, CV_32FC1);

		for (int row = 0; row < hG[0]; ++row) {
			uchar *pixel_ref=grayImage_ref_CV8U.ptr<uchar>(row);
			uchar *pixel_tar=grayImage_ref_CV8U.ptr<uchar>(row);
			float * pixel_depth_ref= inv_depth_ref.ptr<float>(row);

			for (int col = 0; col < wG[0]; ++col) {
				color_ref[row*wG[0]+col]= (float) pixel_ref[col];
				color_tar[row*wG[0]+col]= (float)pixel_tar[col];
				depthMapArray_ref[row*wG[0]+col]=pixel_depth_ref[col];

			}
		}

		newFrame_ref->makeImages(color_ref); // make image_ref pyramid
		newFrame_tar->makeImages(color_tar); // make image_tar pyramid
		depthMap_ref->makeImages(depthMapArray_ref);// make image_ref depth pyramid


		statusMap= new float[wG[0]*hG[0]];
		statusMapB = new bool[wG[0]*hG[0]];

		int setting_desiredImmatureDensity=1500;
//		float densities[] = {0.03,0.05,0.15,0.5,1}; // 不同层取得点密度
		float densities[] = {1,0.5,0.15,0.05,0.03}; // 不同层取得点密度
		//	int npts; // 选择的像素数目
		//	pixelSelector->allowFast = true;
		//	pixelSelector->currentPotential=3;
		//	npts= pixelSelector->makeMaps(newFrame,statusMap, densities[0]*wG[0]*hG[0],1, true,2 );
		//	cout<<"npts:"<<npts;


		int  npts[pyrLevelsUsed];
		// MinimalImageB3 imgShow[pyrLevelsUsed];
		pixelSelector->currentPotential= 3;
		for(int i=0; i>=0; i--)
		{
			if(i!=0){
				options.optimize_depth = false;
			}
			cout<<"\n pyrLevelsUsed:"<<i<<endl;

//			plotImPyr(newFrame_ref, i, "newFrame_ref");
//			plotImPyr(newFrame_tar, i, "newFrame_tar");
//			plotImPyr(depthMap_ref, i, "depthMap_ref");
			npts[i]= pixelSelector->makeMaps(newFrame_ref, statusMap, densities[i]*wG[0]*hG[0], 1, false, 2);
			cout<<"\n npts[i]: "<< npts[i]<<"\n densities[i]*wG[0]*hG[0]:"<<densities[i]*wG[0]*hG[0]<<endl;
			cvvWaitKey(0);


			cv::Mat image_ref(hG[i], wG[i], CV_32FC1);
			memcpy(image_ref.data, newFrame_ref->img_pyr[i], wG[i]*hG[i]*sizeof(float));

			cv::Mat image_tar(hG[i], wG[i], CV_32FC1);
			memcpy(image_tar.data, newFrame_tar->img_pyr[i], wG[i]*hG[i]*sizeof(float));

			cv::Mat depthImg_ref(hG[i], wG[i], CV_32FC1);
			memcpy(depthImg_ref.data, depthMap_ref->img_pyr[i], wG[i]*hG[i]*sizeof(float));


			int idx_EM=0;
			double min_gt_special, max_gt_special;

			image_ref.convertTo(image_ref, CV_64FC1);
			image_tar.convertTo(image_tar, CV_64FC1);
			depthImg_ref.convertTo(depthImg_ref, CV_64FC1);


			while ( idx_EM < 1){
				double  max_n_, min_n_;
				cv::minMaxLoc(deltaMap, &min_n_, &max_n_);
				cout<<"->>>>>>>>>>>>>>>>>show max and min of estimated deltaMap:"<< max_n_ <<","<<min_n_<<endl;
				Mat mask = cv::Mat(deltaMap != deltaMap);
				deltaMap.setTo(1.0, mask);
				if (i==1){
//					cout<<"depthErr(depth_ref_gt, inv_depth_ref).val[0]:"<<depthErr(depth_ref_gt, inv_depth_ref).val[0]<<endl;
//					showScaledImage(depth_ref_NS_before,depth_ref_gt,inv_depth_ref);
				}
				cv::minMaxLoc(depthImg_ref, &min_gt_special, &max_gt_special);
				cout<<"\n show depth_ref min, max:\n"<<min_gt_special<<","<<max_gt_special<<endl;
				Mat inv_depth_ref_for_show= depthImg_ref*(1.0/(max_gt_special-min_gt_special))+(-min_gt_special*(1.0/(max_gt_special-min_gt_special)));
				string depth_ref_name= "inv_depth_ref"+ to_string(i);
//				imshow(depth_ref_name, inv_depth_ref_for_show);
				if (dataLoader->options_.remove_outlier_manually){
//				PhotometricBA(IRef, I, options, Klvl, xi, inv_depth_ref,deltaMap,depth_upper_bound, depth_lower_bound,dataLoader->outlier_mask_big_baseline);
//				//				inv_depth_ref.convertTo(inv_depth_ref, CV_32FC1);
//				//				imwrite("test_inv_depth.exr",inv_depth_ref);
//				//				showScaledImage(depth_ref_gt, inv_depth_ref);
//				//				waitKey(0);
				} else{
//				PhotometricBA(IRef, I, options, Klvl, xi, inv_depth_ref,deltaMap,depth_upper_bound, depth_lower_bound, statusMap, statusMapB);
                PhotometricBA(image_ref, image_tar, options, KG[i], Rotation,Translation, depthImg_ref,deltaMap,depth_upper_bound, depth_lower_bound, statusMap, statusMapB);
//				imshow(depth_ref_name, inv_depth_ref_for_show);
//				waitKey(0);
				}


//			updateDelta(xi,Klvl,image_ref_baseColor,depth_ref,image_ref_metallic ,image_ref_roughness,light_source, deltaMap,newNormalMap,up_new, butt_new);


//			Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K,distanceThres,xi_GT, upper, buttom, deltaMap);
//			Mat showGTdeltaMap=colorMap(deltaMapGT_res, upper, buttom);
//			Mat showESdeltaMap=colorMap(deltaMap, upper, buttom);
//			imshow("show GT deltaMap", showGTdeltaMap);
//			imshow("show ES deltaMap", showESdeltaMap);
//			imwrite("GT_deltaMap.exr",showGTdeltaMap);
//			imwrite("ES_deltaMap.exr",showESdeltaMap);
//
			cout<<"\n show depth_ref min, max:\n"<<min_gt_special<<","<<max_gt_special<<endl;
			cout << "\n Show optimized rotation:\n" << Rotation.matrix()<< "\n Show optimized translation:\n" <<Translation << endl;
			cout << "\n Show Rotational error :"<< rotationErr(xi_GT.rotationMatrix(),  Rotation.matrix()) <<"(degree)."<<"\n Show translational error :" << 100* translationErr(xi_GT.translation(), Translation) <<"(%) "
			     <<"\n Show depth error :"<<depthErr(depth_ref_gt, inv_depth_ref).val[0]<<endl;// !!!!!!!!!!!!!!!!!!!!!!!!

			idx_EM+=1;

			// end of while loop
			}
			cout<<"\nShow current rotation perturbation error :"<< roErr<< "\nShow current translation perturbation error : "<< trErr<<"\nShow current depth perturbation error :"<< depth_Error<<endl;
			waitKey(0);



			// end of for loop
		}





//		int npts;
//		for(int lvl=0; lvl<3; lvl++)
//		{
//			pixelSelector->currentPotential= 3;
//
//			if (lvl == 0)
//				npts =pixelSelector->makeMaps(newFrame, statusMap, densities[lvl] * wG[0] * hG[0], 1, true, 2);
//			else
//				npts = makePixelStatus(newFrame->dIp[lvl], statusMapB, wG[lvl], hG[lvl], densities[lvl] * wG[0] * hG[0]);

//
////			int wl = wG[lvl], hl = hG[lvl]; // 每一层的图像大小
////			int nl = 0;
////			int count=0;
////			for(int y=patternPadding+1;y<hl-patternPadding-2;y++)
////				for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
////				{
////					if((lvl!=0 && statusMapB[x+y*wl]) || (lvl==0 && statusMap[x+y*wl] != 0)){
////				cout<<"show map_out index:"<<x+y*wl<<endl;
////						count++;
////					}
////
////
////				}
////
//		}
//
//		for (int lvl = 3; lvl >= 1; lvl--)
//		{
//			cout<<"==========================show translation=========================:\n"<<xi.translation()<<endl;
//
////			npts =pixelSelector->makeMaps(newFrame, statusMap, densities[lvl] * wG[0] * hG[0], 1, false, 2);
//			cout << "\n Show the value of lvl:" << lvl << endl;
//			Mat IRef, DRef, I, D;
//			Eigen::Matrix3f Klvl, Klvl_ignore;
//			lvl_target = lvl;
//			lvl_ref = lvl;
////			inv_depth_ref;
////			downscale(grayImage_ref, depth_ref, K, lvl_ref, IRef, DRef, Klvl);
////			downscale(grayImage_target, depth_target, K, lvl_target, I, D, Klvl_ignore);
//			downscale(grayImage_ref, inv_depth_ref, K, lvl_ref, IRef, DRef, Klvl);
//			downscale(grayImage_target, depth_target, K, lvl_target, I, D, Klvl_ignore);
//			double min_gt_special, max_gt_special;
//
//			int i=0;
//			while ( i < 2){
//
//				double  max_n_, min_n_;
//				cv::minMaxLoc(deltaMap, &min_n_, &max_n_);
//				cout<<"->>>>>>>>>>>>>>>>>show max and min of estimated deltaMap:"<< max_n_ <<","<<min_n_<<endl;
//				Mat mask = cv::Mat(deltaMap != deltaMap);
//				deltaMap.setTo(1.0, mask);
//				if (i==1){
////					cout<<"depthErr(depth_ref_gt, inv_depth_ref).val[0]:"<<depthErr(depth_ref_gt, inv_depth_ref).val[0]<<endl;
////					showScaledImage(depth_ref_NS_before,depth_ref_gt,inv_depth_ref);
//				}
//				cv::minMaxLoc(inv_depth_ref, &min_gt_special, &max_gt_special);
//				cout<<"\n show depth_ref min, max:\n"<<min_gt_special<<","<<max_gt_special<<endl;
//				Mat inv_depth_ref_for_show= inv_depth_ref*(1.0/(max_gt_special-min_gt_special))+(-min_gt_special*(1.0/(max_gt_special-min_gt_special)));
//				string depth_ref_name= "inv_depth_ref"+ to_string(i);
////				imshow(depth_ref_name, inv_depth_ref_for_show);
//
//
//				if (dataLoader->options_.remove_outlier_manually){
////					PhotometricBA(IRef, I, options, Klvl, xi, inv_depth_ref,deltaMap,depth_upper_bound, depth_lower_bound,dataLoader->outlier_mask_big_baseline);
//					//				inv_depth_ref.convertTo(inv_depth_ref, CV_32FC1);
//					//				imwrite("test_inv_depth.exr",inv_depth_ref);
//					//				showScaledImage(depth_ref_gt, inv_depth_ref);
//					//				waitKey(0);
//				} else{
//					PhotometricBA(IRef, I, options, Klvl, Rotation,Translation, DRef,deltaMap,depth_upper_bound, depth_lower_bound, statusMap, statusMapB);
//					imshow(depth_ref_name, inv_depth_ref_for_show);
//					waitKey(0);
//				}
//
//
////			updateDelta(xi,Klvl,image_ref_baseColor,depth_ref,image_ref_metallic ,image_ref_roughness,light_source, deltaMap,newNormalMap,up_new, butt_new);
//
//
////			Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K,distanceThres,xi_GT, upper, buttom, deltaMap);
////			Mat showGTdeltaMap=colorMap(deltaMapGT_res, upper, buttom);
////			Mat showESdeltaMap=colorMap(deltaMap, upper, buttom);
////			imshow("show GT deltaMap", showGTdeltaMap);
////			imshow("show ES deltaMap", showESdeltaMap);
////			imwrite("GT_deltaMap.exr",showGTdeltaMap);
////			imwrite("ES_deltaMap.exr",showESdeltaMap);
////
//				cout<<"\n show depth_ref min, max:\n"<<min_gt_special<<","<<max_gt_special<<endl;
////			cout << "\n Show initial pose:\n" << xi_GT.rotationMatrix() << "\n Show translation:\n" << xi_GT.translation()<<endl;
//				cout << "\n Show optimized rotation:\n" << xi.rotationMatrix() << "\n Show optimized translation:\n" << xi.translation()<< endl;
//				cout << "\n Show Rotational error :"<< rotationErr(xi_GT.rotationMatrix(), xi.rotationMatrix()) <<"(degree)."<<"\n Show translational error :" << 100* translationErr(xi_GT.translation(), xi.translation()) <<"(%) "
//				     <<"\n Show depth error :"<<depthErr(depth_ref_gt, inv_depth_ref).val[0]<<endl;// !!!!!!!!!!!!!!!!!!!!!!!!
////
////			waitKey(0);
//				i+=1;
//
//			}
//			cout<<"\nShow current rotation perturbation error :"<< roErr<< "\nShow current translation perturbation error : "<< trErr<<"\nShow current depth perturbation error :"<< depth_Error<<endl;
////			waitKey(0);
//
//		}

//		cout<<"npts:"<<npts<< endl;
		// =========================================================Pixel Selector(end here)==============================================================

//		int counter=0;
//		for (int i = 0; i < 300000; ++i) {
//			if (statusMapB[i] || statusMap[i]!=0){
////			cout<<"show map_out index:"<<i<<endl;
//				counter++;
//			}
//		}
//		cout<<"counter:"<<counter<< endl;
	}

	double min_gt_special, max_gt_special;
	cv::minMaxLoc(inv_depth_ref, &min_gt_special, &max_gt_special);
	cout<<"\n show inv_depth_ref min, max:\n"<<min_gt_special<<","<<max_gt_special<<endl;
	Mat inv_depth_ref_for_show= inv_depth_ref*(1.0/(max_gt_special-min_gt_special))+(-min_gt_special*(1.0/(max_gt_special-min_gt_special)));
	string depth_ref_name= "inv_depth_ref";
	imshow(depth_ref_name, inv_depth_ref_for_show);

//	imshow("pfm_depth",inv_depth_ref );

//	savePointCloud(inv_depth_ref,Rotation, Translation);

	waitKey(0);

	for (int lvl = 1; lvl >= 1; lvl--)
	{

		cout << "\n Show the value of lvl:" << lvl << endl;
		Mat IRef, DRef, I, D;
		Eigen::Matrix3f Klvl, Klvl_ignore;
		lvl_target = lvl;
		lvl_ref = lvl;
//			inv_depth_ref;
//			downscale(grayImage_ref, depth_ref, K, lvl_ref, IRef, DRef, Klvl);
//			downscale(grayImage_target, depth_target, K, lvl_target, I, D, Klvl_ignore);
		downscale(grayImage_ref, inv_depth_ref, K, lvl_ref, IRef, DRef, Klvl);
		downscale(grayImage_target, depth_target, K, lvl_target, I, D, Klvl_ignore);
		double min_gt_special, max_gt_special;

		int i=0;
		while ( i < 1){
			double  max_n_, min_n_;
			cv::minMaxLoc(deltaMap, &min_n_, &max_n_);
			cout<<"->>>>>>>>>>>>>>>>>show max and min of estimated deltaMap:"<< max_n_ <<","<<min_n_<<endl;
			Mat mask = cv::Mat(deltaMap != deltaMap);
			deltaMap.setTo(1.0, mask);
			if (i==1){
					cout<<"depthErr(depth_ref_gt, inv_depth_ref).val[0]:"<<depthErr(depth_ref_gt, inv_depth_ref).val[0]<<endl;
					showScaledImage(depth_ref_NS_before,depth_ref_gt,inv_depth_ref);
			}
			cv::minMaxLoc(inv_depth_ref, &min_gt_special, &max_gt_special);
			cout<<"\n show inv_depth_ref min, max:\n"<<min_gt_special<<","<<max_gt_special<<endl;
			Mat inv_depth_ref_for_show= inv_depth_ref*(1.0/(max_gt_special-min_gt_special))+(-min_gt_special*(1.0/(max_gt_special-min_gt_special)));
			string depth_ref_name= "inv_depth_ref"+ to_string(i);
			imshow(depth_ref_name, inv_depth_ref_for_show);
//			cout<<"show the current depth:"<<inv_depth_ref.at<double>(359,470)<<endl;


			if (dataLoader->options_.remove_outlier_manually){
//				PhotometricBA(IRef, I, options, Klvl, xi, inv_depth_ref,deltaMap,depth_upper_bound, depth_lower_bound,dataLoader->outlier_mask_big_baseline);
//				//				inv_depth_ref.convertTo(inv_depth_ref, CV_32FC1);
//				//				imwrite("test_inv_depth.exr",inv_depth_ref);
//				//				showScaledImage(depth_ref_gt, inv_depth_ref);
//				//				waitKey(0);
			} else{
//				PhotometricBA(IRef, I, options, Klvl, xi, inv_depth_ref,deltaMap,depth_upper_bound, depth_lower_bound, statusMap, statusMapB);
				PhotometricBA(IRef, I, options, Klvl, Rotation,Translation, inv_depth_ref,deltaMap,depth_upper_bound, depth_lower_bound, statusMap, statusMapB);

				imshow(depth_ref_name, inv_depth_ref_for_show);
				waitKey(0);
			}


//			updateDelta(xi,Klvl,image_ref_baseColor,depth_ref,image_ref_metallic ,image_ref_roughness,light_source, deltaMap,newNormalMap,up_new, butt_new);


//			Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K,distanceThres,xi_GT, upper, buttom, deltaMap);
//			Mat showGTdeltaMap=colorMap(deltaMapGT_res, upper, buttom);
//			Mat showESdeltaMap=colorMap(deltaMap, upper, buttom);
//			imshow("show GT deltaMap", showGTdeltaMap);
//			imshow("show ES deltaMap", showESdeltaMap);
//			imwrite("GT_deltaMap.exr",showGTdeltaMap);
//			imwrite("ES_deltaMap.exr",showESdeltaMap);
//
			cout<<"\n show depth_ref min, max:\n"<<min_gt_special<<","<<max_gt_special<<endl;
			cout << "\n Show optimized rotation:\n" << Rotation.matrix()<< "\n Show optimized translation:\n" <<Translation << endl;
			cout << "\n Show Rotational error :"<< rotationErr(xi_GT.rotationMatrix(),  Rotation.matrix()) <<"(degree)."<<"\n Show translational error :" << 100* translationErr(xi_GT.translation(), Translation) <<"(%) "
			     <<"\n Show depth error :"<<depthErr(depth_ref_gt, inv_depth_ref).val[0]<<endl;// !!!!!!!!!!!!!!!!!!!!!!!!

			i+=1;

		}
		cout<<"\nShow current rotation perturbation error :"<< roErr<< "\nShow current translation perturbation error : "<< trErr<<"\nShow current depth perturbation error :"<< depth_Error<<endl;
			waitKey(0);

	}

	// tidy up
	delete dataLoader;

	if (options.usePixelSelector){
		delete pixelSelector;
		delete newFrame_ref;
		delete newFrame_tar;
		delete depthMap_ref;
		delete[] statusMap;
		delete[] color_ref;
		delete[] color_tar;
		delete[] depthMapArray_ref;
		delete[] statusMap;
		delete[] statusMapB;
	}


	return 0;
}
