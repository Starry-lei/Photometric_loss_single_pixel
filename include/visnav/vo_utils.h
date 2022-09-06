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

#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // TODO SHEET 5: project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.

  for (const auto& landmark : landmarks) {
    if ((current_pose.inverse() * landmark.second.p).z() > cam_z_threshold) {
      Eigen::Vector2d projected_point;
      projected_point =
          cam->project(current_pose.inverse() *
                       landmark.second.p);  // current_pose.inverse()
      if (projected_point.x() > 0 && projected_point.x() < cam->width() - 1 &&
          projected_point.y() > 0 && projected_point.y() < cam->height() - 1) {
        projected_points.push_back(projected_point);
        projected_track_ids.push_back(landmark.first);
      }
    }
  }

  UNUSED(current_pose);
  UNUSED(cam);
  UNUSED(landmarks);
  UNUSED(cam_z_threshold);
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_threshold,
    const double feature_match_dist_2_best, LandmarkMatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_threshold and feature_match_dist_2_best
  // should be used to filter outliers the same way as in exercise 3. You should
  // fill md.matches with <featureId,trackId> pairs for the successful matches
  // that pass all tests.
  for (size_t i = 0; i < kdl.corners.size(); i++) {
    std::bitset<256> key_point_desc = kdl.corner_descriptors[i];
    int smallest_d = 257;
    int sec_smallest_d = 257;
    int sm_idx = -1;

    for (size_t j = 0; j < projected_points.size(); j++) {
      double dist_p2k = (projected_points[j] - kdl.corners[i]).norm();

      TrackId track_id = projected_track_ids[j];
      TrackId feature_id_min;
      int distance_min = 256;

      if (dist_p2k < match_max_dist_2d) {
        for (const auto& obs : landmarks.at(track_id).obs) {
          FrameCamId fcid = obs.first;
          FeatureId fid = obs.second;

          std::bitset<256> desc =
              feature_corners.at(obs.first).corner_descriptors[obs.second];
          int dist_descri = (desc ^ key_point_desc).count();

          if (dist_descri < distance_min) {
            feature_id_min = obs.second;
            distance_min = dist_descri;
          }
        }
      }

      if (distance_min < smallest_d) {
        sec_smallest_d = smallest_d;
        smallest_d = distance_min;
        sm_idx = projected_track_ids[j];

      } else if (distance_min < sec_smallest_d) {
        sec_smallest_d = distance_min;
      }
    }
    if (smallest_d < feature_match_threshold &&
        smallest_d * feature_match_dist_2_best < sec_smallest_d) {
      md.matches.push_back(std::make_pair(i, sm_idx));
    }
  }

  UNUSED(kdl);
  UNUSED(landmarks);
  UNUSED(feature_corners);
  UNUSED(projected_points);
  UNUSED(projected_track_ids);
  UNUSED(match_max_dist_2d);
  UNUSED(feature_match_threshold);
  UNUSED(feature_match_dist_2_best);
}

void localize_camera(const Sophus::SE3d& current_pose,
                     const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     LandmarkMatchData& md) {
  md.inliers.clear();
  std::vector<int> inliers;
  inliers.clear();
  // default to previous pose if not enough inliers
  md.T_w_c = current_pose;

  if (md.matches.size() < 4) {
    return;
  }

  // TODO SHEET 5: Find the pose (md.T_w_c) and the inliers (md.inliers) using
  // the landmark to keypoints matches and PnP. This should be similar to the
  // localize_camera in exercise 4 but in this exercise we don't explicitly have
  // tracks.

  opengv::bearingVectors_t bearing_vectors_1;
  opengv::points_t points_w;

  for (const auto& match : md.matches) {
    bearing_vectors_1.push_back(
        cam->unproject(kdl.corners[match.first]).normalized());
    points_w.push_back(landmarks.at(match.second).p);
  }

  // use OpenGV’s CentralAbsoluteAdaptertogether with the corresponding RANSAC
  // implementation AbsolutePoseSacProblem
  // adapter for the central absolute pose problem
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vectors_1,
                                                        points_w);

  // Sophus::SE3d T_w_c_pnp;
  // T_w_c_pnp =cameras.at(fcid).T_w_c;

  // create an Ransac-object
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;

  // create an AbsolutePoseSacProblem-object
  // (algorithm is selectable: KNEIP, GAO, or EPNP)
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      problem(new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
          adapter,
          opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP));

  // run ransac

  ransac.sac_model_ = problem;
  double threshold =
      1.0 - cos(atan(sqrt(2.0) * reprojection_error_pnp_inlier_threshold_pixel /
                     500.0));
  ransac.threshold_ = threshold;
  ransac.max_iterations_ = 100;
  ransac.computeModel();

  opengv::bearingVectors_t bearing_vectors_inliers;
  opengv::points_t points_w_inliers;

  for (const auto& inlier : ransac.inliers_) {
    bearing_vectors_inliers.push_back(bearing_vectors_1[inlier]);
    points_w_inliers.push_back(points_w[inlier]);
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter_inlier(
      bearing_vectors_inliers, points_w_inliers);

  adapter_inlier.setR(ransac.model_coefficients_.block(0, 0, 3, 3));
  adapter_inlier.sett(ransac.model_coefficients_.block(0, 3, 3, 1));

  opengv::transformation_t Transform =
      opengv::absolute_pose::optimize_nonlinear(adapter_inlier);

  ransac.sac_model_->selectWithinDistance(Transform, ransac.threshold_,
                                          inliers);

  md.T_w_c =
      Sophus::SE3d(Transform.block(0, 0, 3, 3), Transform.block(0, 3, 3, 1));

  UNUSED(cam);
  UNUSED(kdl);
  UNUSED(landmarks);
  UNUSED(reprojection_error_pnp_inlier_threshold_pixel);
}

void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, const MatchData& md_stereo,
                       const LandmarkMatchData& md, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains feature to landmark
  // matches for the left camera (camera 0). For all inlier feature to landmark
  // matches add the observations to the existing landmarks. If the left
  // camera's feature appears also in md_stereo.inliers, then add both
  // observations. For all inlier stereo observations that were not added to the
  // existing landmarks, triangulate and add new landmarks. Here
  // next_landmark_id is a running index of the landmarks, so after adding a new
  // landmark you should always increase next_landmark_id by 1.

  std::unordered_map<FeatureId, FeatureId> map(md_stereo.inliers.begin(),
                                               md_stereo.inliers.end());

  for (size_t i = 0; i < md.inliers.size(); i++) {
    FeatureId fidl = md.inliers[i].first;
    TrackId track_id = md.inliers[i].second;
    landmarks.at(track_id).obs[fcidl] = fidl;

    if (map.count(fidl) > 0) {
      FeatureId fidr = map.at(fidl);
      landmarks.at(track_id).obs[fcidr] = fidr;
      map.erase(fidl);
    }
  }

  for (const auto& pair : map) {
    FeatureId fid_l = pair.first;
    FeatureId fid_r = pair.second;
    TrackId tid = next_landmark_id;
    next_landmark_id++;

    opengv::bearingVectors_t bearingVector_1;
    opengv::bearingVectors_t bearingVector_2;

    bearingVector_1.push_back(
        calib_cam.intrinsics[0]->unproject(kdl.corners[fid_l]));
    bearingVector_2.push_back(
        calib_cam.intrinsics[1]->unproject(kdr.corners[fid_r]));

    opengv::relative_pose::CentralRelativeAdapter adpater(
        bearingVector_1, bearingVector_2, t_0_1, R_0_1);

    Landmark landmark;
    landmark.p = md.T_w_c * opengv::triangulation::triangulate(adpater, 0);
    landmark.obs[fcidl] = fid_l;
    landmark.obs[fcidr] = fid_r;
    landmarks[tid] = landmark;
  }

  UNUSED(fcidl);
  UNUSED(fcidr);
  UNUSED(kdl);
  UNUSED(kdr);
  UNUSED(calib_cam);
  UNUSED(md_stereo);
  UNUSED(md);
  UNUSED(landmarks);
  UNUSED(next_landmark_id);
  UNUSED(t_0_1);
  UNUSED(R_0_1);
}

void remove_old_keyframes(const FrameCamId fcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(fcidl.frame_id);

  // TODO SHEET 5: Remove old cameras and observations if the number of keyframe
  // pairs (left and right image is a pair) is larger than max_num_kfs. The ids
  // of all the keyframes that are currently in the optimization should be
  // stored in kf_frames. Removed keyframes should be removed from cameras and
  // landmarks with no left observations should be moved to old_landmarks.

  std::vector<TrackId> landmarks_remove;
  while (int(kf_frames.size()) > max_num_kfs) {
    FrameCamId fcid_l, fcid_r;
    fcid_l.cam_id = 0;
    fcid_r.cam_id = 1;
    fcid_l.frame_id = *kf_frames.begin();
    fcid_r.frame_id = *kf_frames.begin();

    cameras.erase(fcid_l);
    cameras.erase(fcid_r);

    for (auto& landmark : landmarks) {
      landmark.second.obs.erase(fcid_l);
      landmark.second.obs.erase(fcid_r);
      if (landmark.second.obs.size() == 0) {
        old_landmarks.emplace(landmark);
        landmarks_remove.push_back(landmark.first);
      }
    }

    kf_frames.erase(kf_frames.begin());
  }

  for (size_t i = 0; i < landmarks_remove.size(); i++) {
    landmarks.erase(landmarks_remove[i]);
  }

  UNUSED(max_num_kfs);
  UNUSED(cameras);
  UNUSED(landmarks);
  UNUSED(old_landmarks);
}
}  // namespace visnav
