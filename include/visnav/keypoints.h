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

#include <pangolin/image/managed_image.h>

#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>

#include <map>
#include <visnav/common_types.h>
using std::map;
namespace visnav {

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

typedef std::bitset<256> Descriptor;

char pattern_31_x_a[256] = {
    8,   4,   -11, 7,   2,   1,   -2,  -13, -13, 10,  -13, -11, 7,   -4,  -13,
    -9,  12,  -3,  -6,  11,  4,   5,   3,   -8,  -2,  -13, -7,  -4,  -10, 5,
    5,   1,   9,   4,   2,   -4,  -8,  4,   0,   -13, -3,  -6,  8,   0,   7,
    -13, 10,  -6,  10,  -13, -13, 3,   5,   -1,  3,   2,   -13, -13, -13, -7,
    6,   -9,  -2,  -12, 3,   -7,  -3,  2,   -11, -1,  5,   -4,  -9,  -12, 10,
    7,   -7,  -4,  7,   -7,  -13, -3,  7,   -13, 1,   2,   -4,  -1,  7,   1,
    9,   -1,  -13, 7,   12,  6,   5,   2,   3,   2,   9,   -8,  -11, 1,   6,
    2,   6,   3,   7,   -11, -10, -5,  -10, 8,   4,   -10, 4,   -2,  -5,  7,
    -9,  -5,  8,   -9,  1,   7,   -2,  11,  -12, 3,   5,   0,   -9,  0,   -1,
    5,   3,   -13, -5,  -4,  6,   -7,  -13, 1,   4,   -2,  2,   -2,  4,   -6,
    -3,  7,   4,   -13, 7,   7,   -7,  -8,  -13, 2,   10,  -6,  8,   2,   -11,
    -12, -11, 5,   -2,  -1,  -13, -10, -3,  2,   -9,  -4,  -4,  -6,  6,   -13,
    11,  7,   -1,  -4,  -7,  -13, -7,  -8,  -5,  -13, 1,   1,   9,   5,   -1,
    -9,  -1,  -13, 8,   2,   7,   -10, -10, 4,   3,   -4,  5,   4,   -9,  0,
    -12, 3,   -10, 8,   -8,  2,   10,  6,   -7,  -3,  -1,  -3,  -8,  4,   2,
    6,   3,   11,  -3,  4,   2,   -10, -13, -13, 6,   0,   -13, -9,  -13, 5,
    2,   -1,  9,   11,  3,   -1,  3,   -13, 5,   8,   7,   -10, 7,   9,   7,
    -1};

char pattern_31_y_a[256] = {
    -3,  2,   9,   -12, -13, -7,  -10, -13, -3,  4,   -8,  7,   7,   -5,  2,
    0,   -6,  6,   -13, -13, 7,   -3,  -7,  -7,  11,  12,  3,   2,   -12, -12,
    -6,  0,   11,  7,   -1,  -12, -5,  11,  -8,  -2,  -2,  9,   12,  9,   -5,
    -6,  7,   -3,  -9,  8,   0,   3,   7,   7,   -10, -4,  0,   -7,  3,   12,
    -10, -1,  -5,  5,   -10, -7,  -2,  9,   -13, 6,   -3,  -13, -6,  -10, 2,
    12,  -13, 9,   -1,  6,   11,  7,   -8,  -7,  -3,  -6,  3,   -13, 1,   -1,
    1,   -9,  -13, 7,   -5,  3,   -13, -12, 8,   6,   -12, 4,   12,  12,  -9,
    3,   3,   -3,  8,   -5,  11,  -8,  5,   -1,  -6,  12,  -2,  0,   -8,  -6,
    -13, -13, -8,  -11, -8,  -4,  1,   -6,  -9,  7,   5,   -4,  12,  7,   2,
    11,  5,   -4,  9,   -7,  5,   6,   6,   -10, 1,   -2,  -12, -13, 1,   -10,
    -13, 5,   -2,  9,   1,   -8,  -4,  11,  6,   4,   -5,  -5,  -3,  -12, -2,
    -13, 0,   -3,  -13, -8,  -11, -2,  9,   -3,  -13, 6,   12,  -11, -3,  11,
    11,  -5,  12,  -8,  1,   -12, -2,  5,   -1,  7,   5,   0,   12,  -8,  11,
    -3,  -10, 1,   -11, -13, -13, -10, -8,  -6,  12,  2,   -13, -13, 9,   3,
    1,   2,   -10, -13, -12, 2,   6,   8,   10,  -9,  -13, -7,  -2,  2,   -5,
    -9,  -1,  -1,  0,   -11, -4,  -6,  7,   12,  0,   -1,  3,   8,   -6,  -9,
    7,   -6,  5,   -3,  0,   4,   -6,  0,   8,   9,   -4,  4,   3,   -7,  0,
    -6};

char pattern_31_x_b[256] = {
    9,   7,  -8, 12,  2,   1,  -2,  -11, -12, 11,  -8,  -9,  12,  -3,  -12, -7,
    12,  -2, -4, 12,  5,   10, 6,   -6,  -1,  -8,  -5,  -3,  -6,  6,   7,   4,
    11,  4,  4,  -2,  -7,  9,  1,   -8,  -2,  -4,  10,  1,   11,  -11, 12,  -6,
    12,  -8, -8, 7,   10,  1,  5,   3,   -13, -12, -11, -4,  12,  -7,  0,   -7,
    8,   -4, -1, 5,   -5,  0,  5,   -4,  -9,  -8,  12,  12,  -6,  -3,  12,  -5,
    -12, -2, 12, -11, 12,  3,  -2,  1,   8,   3,   12,  -1,  -10, 10,  12,  7,
    6,   2,  4,  12,  10,  -7, -4,  2,   7,   3,   11,  8,   9,   -6,  -5,  -3,
    -9,  12, 6,  -8,  6,   -2, -5,  10,  -8,  -5,  9,   -9,  1,   9,   -1,  12,
    -6,  7,  10, 2,   -5,  2,  1,   7,   6,   -8,  -3,  -3,  8,   -6,  -5,  3,
    8,   2,  12, 0,   9,   -3, -1,  12,  5,   -9,  8,   7,   -7,  -7,  -12, 3,
    12,  -6, 9,  2,   -10, -7, -10, 11,  -1,  0,   -12, -10, -2,  3,   -4,  -3,
    -2,  -4, 6,  -5,  12,  12, 0,   -3,  -6,  -8,  -6,  -6,  -4,  -8,  5,   10,
    10,  10, 1,  -6,  1,   -8, 10,  3,   12,  -5,  -8,  8,   8,   -3,  10,  5,
    -4,  3,  -6, 4,   -10, 12, -6,  3,   11,  8,   -6,  -3,  -1,  -3,  -8,  12,
    3,   11, 7,  12,  -3,  4,  2,   -8,  -11, -11, 11,  1,   -9,  -6,  -8,  8,
    3,   -1, 11, 12,  3,   0,  4,   -10, 12,  9,   8,   -10, 12,  10,  12,  0};

char pattern_31_y_b[256] = {
    5,   -12, 2,   -13, 12,  6,   -4,  -8,  -9,  9,   -9,  12,  6,   0,  -3,
    5,   -1,  12,  -8,  -8,  1,   -3,  12,  -2,  -10, 10,  -3,  7,   11, -7,
    -1,  -5,  -13, 12,  4,   7,   -10, 12,  -13, 2,   3,   -9,  7,   3,  -10,
    0,   1,   12,  -4,  -12, -4,  8,   -7,  -12, 6,   -10, 5,   12,  8,  7,
    8,   -6,  12,  5,   -13, 5,   -7,  -11, -13, -1,  2,   12,  6,   -4, -3,
    12,  5,   4,   2,   1,   5,   -6,  -7,  -12, 12,  0,   -13, 9,   -6, 12,
    6,   3,   5,   12,  9,   11,  10,  3,   -6,  -13, 3,   9,   -6,  -8, -4,
    -2,  0,   -8,  3,   -4,  10,  12,  0,   -6,  -11, 7,   7,   12,  2,  12,
    -8,  -2,  -13, 0,   -2,  1,   -4,  -11, 4,   12,  8,   8,   -13, 12, 7,
    -9,  -8,  9,   -3,  -12, 0,   12,  -2,  10,  -4,  -13, 12,  -6,  3,  -5,
    1,   -11, -7,  -5,  6,   6,   1,   -8,  -8,  9,   3,   7,   -8,  8,  3,
    -9,  -5,  8,   12,  9,   -5,  11,  -13, 2,   0,   -10, -7,  9,   11, 5,
    6,   -2,  7,   -2,  7,   -13, -8,  -9,  5,   10,  -13, -13, -1,  -9, -13,
    2,   12,  -10, -6,  -6,  -9,  -7,  -13, 5,   -13, -3,  -12, -1,  3,  -9,
    1,   -8,  9,   12,  -5,  7,   -8,  -12, 5,   9,   5,   4,   3,   12, 11,
    -13, 12,  4,   6,   12,  1,   1,   1,   -13, -13, 4,   -2,  -3,  -2, 10,
    -9,  -1,  -2,  -8,  5,   10,  5,   5,   11,  -6,  -12, 9,   4,   -2, -2,
    -11};

void detectKeypoints(const pangolin::ManagedImage<uint8_t> &img_raw,
                     KeypointsData &kd, int num_features) {
  cv::Mat image(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);

  std::vector<cv::Point2f> points;
  goodFeaturesToTrack(image, points, num_features, 0.01, 8);

  kd.corners.clear();
  kd.corner_angles.clear();
  kd.corner_descriptors.clear();

  for (size_t i = 0; i < points.size(); i++) {
    if (img_raw.InBounds(points[i].x, points[i].y, EDGE_THRESHOLD)) {
      kd.corners.emplace_back(points[i].x, points[i].y);
    }
  }
}

void computeAngles(const pangolin::ManagedImage<uint8_t> &img_raw,
                   KeypointsData &kd, bool rotate_features) {
  kd.corner_angles.resize(kd.corners.size());

  int HALF_PATCH_SIZE = 15;
  // cv::Mat image(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);

  for (size_t i = 0; i < kd.corners.size(); i++) {
    const Eigen::Vector2d &p = kd.corners[i];

    const int cx = p[0];
    const int cy = p[1];

    double angle = 0;

    if (rotate_features) {
      // TODO SHEET 3: compute angle

      double m01 = 0, m10 = 0;
      for (int i = -HALF_PATCH_SIZE; i < HALF_PATCH_SIZE + 1; i++) {
        for (int j = -HALF_PATCH_SIZE; j < HALF_PATCH_SIZE + 1; j++) {
          if (i * i + j * j <= 15 * 15) {
            m01 += j * img_raw(cx + i, cy + j);
            m10 += i * img_raw(cx + i, cy + j);
          }
        }
      }

      angle = atan2(m01, m10);

      UNUSED(img_raw);
      UNUSED(cx);
      UNUSED(cy);
    }

    kd.corner_angles[i] = angle;
  }
}

void computeDescriptors(const pangolin::ManagedImage<uint8_t> &img_raw,
                        KeypointsData &kd) {
  kd.corner_descriptors.resize(kd.corners.size());

  for (size_t i = 0; i < kd.corners.size(); i++) {
    std::bitset<256> descriptor;

    const Eigen::Vector2d &p = kd.corners[i];
    const double angle = kd.corner_angles[i];

    const int cx = p[0];
    const int cy = p[1];

    // TODO SHEET 3: compute descriptor
    Eigen::Matrix<double, 2, 2> rotation;
    rotation << cos(angle), -sin(angle), sin(angle), cos(angle);

    for (size_t i = 0; i < 256; i++) {
      Eigen::Vector2i p_a_prime;
      Eigen::Vector2d p_a;
      Eigen::Vector2i p_b_prime;
      Eigen::Vector2d p_b;
      p_a << (double)pattern_31_x_a[i], (double)pattern_31_y_a[i];

      // std::cout<<"show p_a" <<p_a(1)<<std::endl;;
      p_b << (double)pattern_31_x_b[i], (double)pattern_31_y_b[i];

      Eigen::Vector2d temp_a = rotation * p_a; //?

      p_a_prime << round(temp_a(0)), round(temp_a(1));

      Eigen::Vector2d temp_b = rotation * p_b;
      p_b_prime << round(temp_b(0)), round(temp_b(1));

      if (img_raw(cx + p_a_prime(0), cy + p_a_prime(1)) <
          img_raw(cx + p_b_prime(0), cy + p_b_prime(1))) {
        descriptor[i] = 1;
      } else {
        descriptor[i] = 0;
      }
    }

    UNUSED(img_raw);
    UNUSED(angle);
    UNUSED(cx);
    UNUSED(cy);

    kd.corner_descriptors[i] = descriptor;
  }
}

void detectKeypointsAndDescriptors(
    const pangolin::ManagedImage<uint8_t> &img_raw, KeypointsData &kd,
    int num_features, bool rotate_features) {
  detectKeypoints(img_raw, kd, num_features);
  computeAngles(img_raw, kd, rotate_features);
  computeDescriptors(img_raw, kd);
}

// bool sortbyth(const std::tuple<int, int, int>& a,
//               const std::tuple<int, int, int>& b)
// {
//     return (std::get<2>(a) < std::get<2>(b));
// }
void matchDescriptors(const std::vector<std::bitset<256>> &corner_descriptors_1,
                      const std::vector<std::bitset<256>> &corner_descriptors_2,
                      std::vector<std::pair<int, int>> &matches, int threshold,
                      double dist_2_best) {
  matches.clear();

  // TODO SHEET 3: match features

  std::map<int, int> matches_P2Q;
  std::map<int, int> matches_Q2P;
  for (size_t p_i = 0; p_i < corner_descriptors_1.size(); p_i++) {
    int smallest_d = 257;
    int sec_smallest_d = 257;
    int sm_idx = -1;
    for (size_t q_j = 0; q_j < corner_descriptors_2.size(); q_j++) {
      int Hamming_dist = (corner_descriptors_1[p_i] ^ corner_descriptors_2[q_j])
                             .count(); // the number of bits that are different

      if (Hamming_dist < smallest_d) {
        int temp = smallest_d;
        smallest_d = Hamming_dist;
        sm_idx = q_j;
        sec_smallest_d = temp;
      } else if (Hamming_dist < sec_smallest_d) {
        sec_smallest_d = Hamming_dist;
      }
    }
    if (smallest_d < threshold && sec_smallest_d >= smallest_d * dist_2_best) {
      // match[pi] = sm_idx

      matches_P2Q.emplace(p_i, sm_idx);
    }
  }

  // std::cout << "what is the value of matches_P2Q:" << matches_P2Q.size()
  //           << std::endl;
  // sort(matches_P2Q.begin(), matches_P2Q.end(), sortbyth);
  for (size_t q_i = 0; q_i < corner_descriptors_2.size(); q_i++) {
    int smallest_d = 257;
    int sec_smallest_d = 257;
    int sm_idx = -1;
    for (size_t p_j = 0; p_j < corner_descriptors_1.size(); p_j++) {
      int Hamming_dist = (corner_descriptors_2[q_i] ^ corner_descriptors_1[p_j])
                             .count(); // the number of bits that are different

      if (Hamming_dist < smallest_d) {
        int temp = smallest_d;
        smallest_d = Hamming_dist;
        sm_idx = p_j;
        sec_smallest_d = temp;

      } else if (Hamming_dist < sec_smallest_d) {
        sec_smallest_d = Hamming_dist;
      }
    }

    if (smallest_d < threshold && sec_smallest_d >= smallest_d * dist_2_best) {
      matches_Q2P.emplace(q_i, sm_idx);
    }
  }

  // std::cout << "what is the value of matches_Q2P:" << matches_Q2P.size()
  //           << std::endl;

  for (auto ma = matches_P2Q.begin(); ma != matches_P2Q.end(); ++ma) {
    if (matches_Q2P.count(ma->second) &&
        ma->first == matches_Q2P.at(ma->second)) {
      matches.emplace_back(ma->first, ma->second);
    }
  }

  UNUSED(corner_descriptors_1);
  UNUSED(corner_descriptors_2);
  UNUSED(matches);
  UNUSED(threshold);
  UNUSED(dist_2_best);
}

} // namespace visnav
