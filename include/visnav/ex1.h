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

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 3> I = Eigen::Matrix<T, 3, 3>::Identity();
  Eigen::Matrix<T, 3, 3> expmap_of_xi;
  T theta = xi.norm();  // l2 norm
  if (theta == T(0)) {
    return I;
  }
  Eigen::Matrix<T, 3, 1> v = xi / theta;
  T a = v(0);
  T b = v(1);
  T c = v(2);
  Eigen::Matrix<T, 3, 3> v_hat;
  v_hat << 0, -c, b, c, 0, -a, -b, a, 0;
  expmap_of_xi = I + sin(theta) * v_hat + (1 - cos(theta)) * v_hat * v_hat;
  UNUSED(xi);
  return Eigen::Matrix<T, 3, 3>(expmap_of_xi);
}
// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 1> so3_w;
  Eigen::Matrix<T, 3, 1> so3_w_temp;
  so3_w_temp << (mat(2, 1) - mat(1, 2)), (mat(0, 2) - mat(2, 0)),
      (mat(1, 0) - mat(0, 1));
  T theta = acos((T(1) / T(2)) * (mat.trace() - T(1)));
  if (theta == T(0)) {
    return Eigen::Matrix<T, 3, 1>::Zero();
  }
  so3_w = (T(1) / T(2)) * (theta / sin(theta)) * so3_w_temp;
  UNUSED(mat);
  return Eigen::Matrix<T, 3, 1>(so3_w);
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 3> I = Eigen::Matrix<T, 3, 3>::Identity();
  Eigen::Matrix<T, 4, 4> expmap_of_xi;
  Eigen::Matrix<T, 3, 1> translation;
  Eigen::Matrix<T, 3, 1> rotation;
  translation = xi.head(3);
  rotation = xi.tail(3);
  T theta = rotation.norm();  // l2 norm
  if (theta == T(0)) {
    expmap_of_xi << T(1), T(0), T(0), translation(0), T(0), T(1), T(0),
        translation(1), T(0), T(0), T(1), translation(2), T(0), T(0), T(0),
        T(1);
    return expmap_of_xi;
  }
  Eigen::Matrix<T, 3, 1> axis = rotation / theta;
  T a = axis(0);
  T b = axis(1);
  T c = axis(2);
  Eigen::Matrix<T, 3, 3> axis_hat;
  axis_hat << 0, -c, b, c, 0, -a, -b, a, 0;
  Eigen::Matrix<T, 3, 3> J;
  J = I + ((1 - cos(theta)) / theta) * axis_hat +
      ((theta - sin(theta)) / theta) * axis_hat * axis_hat;
  Eigen::Matrix<T, 3, 1> Jv = J * translation;
  Eigen::Matrix<T, 3, 3> SO3_R =
      I + sin(theta) * axis_hat + (1 - cos(theta)) * axis_hat * axis_hat;
  expmap_of_xi << SO3_R, Jv, (Eigen::Matrix<T, 3, 1>::Zero()).transpose(), T(1);
  UNUSED(xi);
  return Eigen::Matrix<T, 4, 4>(expmap_of_xi);
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 6, 1> se3;
  Eigen::Matrix<T, 3, 3> R;
  Eigen::Matrix<T, 3, 1> t;
  Eigen::Matrix<T, 3, 1> v;
  Eigen::Matrix<T, 3, 3> logR;
  Eigen::Matrix<T, 3, 3> J_rev;
  Eigen::Matrix<T, 3, 3> I = Eigen::Matrix<T, 3, 3>::Identity();
  T theta;
  Eigen::Matrix<T, 3, 1> so3_w;
  Eigen::Matrix<T, 3, 1> so3_w_temp;
  Eigen::Matrix<T, 4, 1> vec = mat.rightCols(1);
  R.block(0, 0, 3, 3) = mat.block(0, 0, 3, 3);
  t.col(0) = vec.head(3);
  so3_w_temp << (R(2, 1) - R(1, 2)), (R(0, 2) - R(2, 0)), (R(1, 0) - R(0, 1));
  theta = acos((T(1) / T(2)) * (R.trace() - T(1)));
  if (theta == T(0)) {
    se3.head(3) = t.col(0);
    se3.tail(3) = Eigen::Matrix<T, 3, 1>::Zero();
    return Eigen::Matrix<T, 6, 1>(se3);
  }
  so3_w = (T(1) / T(2)) * (theta / sin(theta)) * so3_w_temp;
  Eigen::Matrix<T, 3, 1> axis = so3_w / theta;
  T a = axis(0);
  T b = axis(1);
  T c = axis(2);
  Eigen::Matrix<T, 3, 3> axis_hat;
  axis_hat << 0, -c, b, c, 0, -a, -b, a, 0;
  J_rev = I - (T(1) / T(2)) * theta * axis_hat +
          (T(1) - ((T(1) + cos(theta)) * theta / (T(2) * sin(theta)))) *
              axis_hat * axis_hat;
  v = J_rev * t;
  se3.head(3) = v.col(0);
  se3.tail(3) = so3_w.col(0);
  UNUSED(mat);
  return Eigen::Matrix<T, 6, 1>(se3);
}

}  // namespace visnav
