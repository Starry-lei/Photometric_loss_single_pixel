
#include <PCLOpt.h>



void comp_accurate_normals(std::vector<Eigen::Vector3d> cloud_eigen, cv::Mat & init_normal_map)
{
	// convert format
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->points.resize(cloud_eigen.size());
	for(int i=0; i<cloud_eigen.size(); i++)
	{
		cloud->points[i].getVector3fMap() = Eigen::Vector3f(cloud_eigen[i](0), cloud_eigen[i](1), cloud_eigen[i](2)).cast<float>();

	}

	// resample
	resamplePts_and_compNormal(cloud, cloud_eigen, init_normal_map);

	std::cout << "---------" << std::endl;
}


void resamplePts_and_compNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_init, std::vector<Eigen::Vector3d> cloud_eigen, cv::Mat& init_normal_map)
{

	// Create a KD-Tree
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

	// Output has the PointNormal type in order to store the normals calculated by MLS
	pcl::PointCloud<pcl::PointNormal> mls_points;

	// Init object (second point type is for the normals, even if unused)
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;

	mls.setComputeNormals (true);

	// Set parameters
	mls.setInputCloud (cloud_init);
	mls.setPolynomialOrder (2);
	mls.setSearchMethod (tree);
//    mls.setSearchRadius (0.1); // 304511 pts
//    mls.setSearchRadius (0.15); // 306776 pts
//        mls.setSearchRadius (0.2); // 307073 pts
	mls.setSearchRadius (0.5); // 307200 pts



	// Reconstruct
	mls.process (mls_points);

//	std::cout << "number of new pts:" << std::endl;
//	std::cout << mls_points.size() << std::endl;


	int n_eigen_current = 0;

	for(int nIndex = 0; nIndex < (mls_points).points.size(); nIndex++)
	{

		double n_x = (mls_points).points[nIndex].normal_x;
		double n_y = (mls_points).points[nIndex].normal_y;
		double n_z = (mls_points).points[nIndex].normal_z;

		Eigen::Vector3d normal_new(n_x, n_y, n_z);
		normal_new = normal_new.normalized();

		Eigen::Vector3d principal_axis(0, 0, 1);
		if(normal_new.dot(principal_axis)>0)
		{
			normal_new = -normal_new;
		}

		//
		double pt_x = (mls_points).points[nIndex].x;
		double pt_y = (mls_points).points[nIndex].y;
		double pt_z = (mls_points).points[nIndex].z;

		Eigen::Vector3d pt_mls(pt_x, pt_y, pt_z);

		bool is_match = false;
		n_eigen_current = iterate_all_pts(n_eigen_current, cloud_eigen, pt_mls, // assign pt_mls's normal to eigen map's pixel
		                                  is_match);

		if(is_match == false)
		{
			std::cout << "use orig value !!!!!!!!" << std::endl;
			continue;
		}


//        std::cout << "successful matching" << std::endl;

		int row_id = trunc(n_eigen_current/640); // 640 pixels in a row
		int col_id = (n_eigen_current - row_id*640) % 640;

//		init_normal_map.at<cv::Vec3d>(row_id, col_id)[0] = normal_new(2)*0.5+0.5;
//		init_normal_map.at<cv::Vec3d>(row_id, col_id)[1] = normal_new(1)*0.5+0.5;
//		init_normal_map.at<cv::Vec3d>(row_id, col_id)[2] = normal_new(0)*0.5+0.5;
		init_normal_map.at<cv::Vec3d>(row_id, col_id)[0] = normal_new(0);
		init_normal_map.at<cv::Vec3d>(row_id, col_id)[1] = normal_new(1);
		init_normal_map.at<cv::Vec3d>(row_id, col_id)[2] = normal_new(2);

		n_eigen_current = n_eigen_current + 1;

	}

	std::cout << "====== complete! =======" << std::endl;
//	mls_points->clear();

//	cv::imshow("img3", init_normal_map);
//	cv::waitKey(0);
//	int a=0;


}


int iterate_all_pts(int n_eigen_current, const std::vector<Eigen::Vector3d>& cloud_eigen, Eigen::Vector3d pt_mls,
                    bool& is_match) // n_eigen is a bigger one
{
	int match_n_eigen = 0;

	for(int n_eigen = n_eigen_current; n_eigen < cloud_eigen.size(); n_eigen++)
	{
		Eigen::Vector3d pt_eigen = cloud_eigen[n_eigen];



		Eigen::Vector3d vec_diff = pt_eigen - pt_mls;
		if(vec_diff.norm() < 0.1)
//        if(vec_diff.norm() < 0.05)
		{
			match_n_eigen = n_eigen;
			is_match = true;
			break;
		}

	}

	if(is_match == false)
	{
		std::cout << "fail to match !!!!!!!!" << std::endl;
		match_n_eigen = n_eigen_current;
	}



	return match_n_eigen;
}