/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

void computeEssential(const Sophus::SE3d &T_0_1, Eigen::Matrix3d &E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();
  Eigen::Vector3d t_0_1_n = t_0_1.normalized();
  // TODO SHEET 3: compute essential matrix
  Eigen::Matrix3d t_skew_symmetric;
  t_skew_symmetric << (double)0, -t_0_1_n(2), t_0_1_n(1), t_0_1_n(2), (double)0,
      -t_0_1_n(0), -t_0_1_n(1), t_0_1_n(0), (double)0;

  E = t_skew_symmetric * R_0_1;

  UNUSED(E);
  UNUSED(t_0_1);
  UNUSED(R_0_1);
}

void findInliersEssential(const KeypointsData &kd1, const KeypointsData &kd2,
                          const std::shared_ptr<AbstractCamera<double>> &cam1,
                          const std::shared_ptr<AbstractCamera<double>> &cam2,
                          const Eigen::Matrix3d &E,
                          double epipolar_error_threshold, MatchData &md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    // TODO SHEET 3: determine inliers and store in md.inliers
    Eigen::Vector3d x1 = cam1->unproject(p0_2d);
    Eigen::Vector3d x2 = cam2->unproject(p1_2d);
    if (abs(x1.transpose() * E * x2) < epipolar_error_threshold) {
      md.inliers.emplace_back(md.matches[j].first, md.matches[j].second);
    }

    UNUSED(cam1);
    UNUSED(cam2);
    UNUSED(E);
    UNUSED(epipolar_error_threshold);
    UNUSED(p0_2d);
    UNUSED(p1_2d);
  }
}

void findInliersRansac(const KeypointsData &kd1, const KeypointsData &kd2,
                       const std::shared_ptr<AbstractCamera<double>> &cam1,
                       const std::shared_ptr<AbstractCamera<double>> &cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData &md) {
  md.inliers.clear();
  md.T_i_j = Sophus::SE3d();

  // TODO SHEET 3: Run RANSAC with using opengv's CentralRelativePose and store
  // the final inlier indices in md.inliers and the final relative pose in
  // md.T_i_j (normalize translation). If the number of inliers is smaller than
  // ransac_min_inliers, leave md.inliers empty. Note that if the initial RANSAC
  // was successful, you should do non-linear refinement of the model parameters
  // using all inliers, and then re-estimate the inlier set with the refined
  // model parameters.

  opengv::bearingVectors_t bearing_vectors_1;
  opengv::bearingVectors_t bearing_vectors_2;
  for (size_t i = 0; i < md.matches.size(); i++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[i].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[i].second];

    Eigen::Vector3d x1 = (cam1->unproject(p0_2d)).normalized();
    Eigen::Vector3d x2 = (cam2->unproject(p1_2d)).normalized();
    bearing_vectors_1.push_back(x1);
    bearing_vectors_2.push_back(x2);
  }

  // create the central relative adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(bearing_vectors_1,
                                                        bearing_vectors_2);
  // create a RANSAC object
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;

  // create a CentralRelativePoseSacProblem
  // (set algorithm to STEWENIUS, NISTER, SEVENPT, or EIGHTPT)
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter, opengv::sac_problems::relative_pose::
                               CentralRelativePoseSacProblem::NISTER));

  // run ransac
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  ransac.max_iterations_ = 150;
  ransac.computeModel();
  // get the result
  opengv::transformation_t best_transformation = ransac.model_coefficients_;

  // non-linear optimization (using all correspondences)

  Eigen::Matrix3d R = best_transformation.topLeftCorner(3, 3);
  Eigen::Vector3d t = best_transformation.topRightCorner(3, 1);

  adapter.setR12(R);
  adapter.sett12(t);

  opengv::transformation_t nonlinear_transformation =
      opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);

  ransac.sac_model_->selectWithinDistance(ransac.model_coefficients_,
                                          ransac.threshold_, ransac.inliers_);
  Sophus::SE3d test(nonlinear_transformation.topLeftCorner(3, 3),
                    nonlinear_transformation.topRightCorner(3, 1));
  md.T_i_j = test;

  for (size_t i = 0; i < ransac.inliers_.size(); i++) {
    md.inliers.emplace_back(md.matches[ransac.inliers_[i]].first,
                            md.matches[ransac.inliers_[i]].second);
  }

  if ((int)(md.matches.size()) < ransac_min_inliers) {
    md.matches.clear();
  }

  UNUSED(kd1);
  UNUSED(kd2);
  UNUSED(cam1);
  UNUSED(cam2);
  UNUSED(ransac_thresh);
  UNUSED(ransac_min_inliers);
}
} // namespace visnav
