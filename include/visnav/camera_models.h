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

#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

template <typename Scalar>
class AbstractCamera;

template <typename Scalar>
class PinholeCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  PinholeCamera() = default;
  PinholeCamera(const VecN& p) : param(p) {}

  static PinholeCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0, 0, 0, 0;
    PinholeCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "pinhole"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    // TODO SHEET 2: implement camera model
    if (z > Scalar(0)) {
      res << fx * x / z + cx, fy * y / z + cy;
    }
    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(x);
    UNUSED(y);
    UNUSED(z);

    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    Vec3 res, res_temp;

    // TODO SHEET 2: implement camera model...........done
    const Scalar& u = p[0];
    const Scalar& v = p[1];

    const Scalar& mx = (u - cx) / fx;
    const Scalar& my = (v - cy) / fy;

    res_temp << mx, my, Scalar(1);
    res = Scalar(1) / res_temp.norm() * res_temp;
    UNUSED(p);
    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar = double>
class ExtendedUnifiedCamera : public AbstractCamera<Scalar> {
 public:
  // NOTE: For convenience for serialization and handling different camera
  // models in ceres functors, we use the same parameter vector size for all of
  // them, even if that means that for some certain entries are unused /
  // constant 0.
  static constexpr int N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vec4;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  ExtendedUnifiedCamera() = default;
  ExtendedUnifiedCamera(const VecN& p) : param(p) {}

  static ExtendedUnifiedCamera getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 500, 0.5 * 500, 319.5, 239.5, 0.51231234, 0.9, 0, 0;
    ExtendedUnifiedCamera res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }
  const Scalar* data() const { return param.data(); }

  static const std::string getName() { return "eucm"; }
  std::string name() const { return getName(); }

  inline Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& alpha = param[4];
    const Scalar& beta = param[5];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    // TODO SHEET 2: implement camera model

    // d=p.norm();

    if (z > Scalar(0)) {
      const Scalar& d = sqrt(beta * (x * x + y * y) + z * z);
      const Scalar& deno = alpha * d + (Scalar(1) - alpha) * z;
      res << fx * x / deno + cx, fy * y / deno + cy;
    }

    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(alpha);
    UNUSED(beta);
    UNUSED(x);
    UNUSED(y);
    UNUSED(z);

    return res;
  }

  Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& alpha = param[4];
    const Scalar& beta = param[5];

    Vec3 res, res_temp;

    // TODO SHEET 2: implement camera model

    const Scalar& u = p[0];
    const Scalar& v = p[1];
    const Scalar& mx = (u - cx) / fx;
    const Scalar& my = (v - cy) / fy;

    const Scalar& r_square = mx * mx + my * my;

    const Scalar& mz =
        (Scalar(1) - beta * alpha * alpha * r_square) /
        (alpha * sqrt(Scalar(1) -
                      (Scalar(2) * alpha - Scalar(1)) * beta * r_square) +
         (Scalar(1) - alpha));
    res_temp << mx, my, mz;
    res = Scalar(1) / res_temp.norm() * res_temp;

    UNUSED(p);
    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(alpha);
    UNUSED(beta);

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar>
class DoubleSphereCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  DoubleSphereCamera() = default;
  DoubleSphereCamera(const VecN& p) : param(p) {}

  static DoubleSphereCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0.5 * -0.150694, 0.5 * 1.48785, 0,
        0;
    DoubleSphereCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }
  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "ds"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& xi = param[4];
    const Scalar& alpha = param[5];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    // TODO SHEET 2: implement camera model
    if (z > Scalar(0)) {
      const Scalar& d1 = p.norm();
      const Scalar& d2 = sqrt(x * x + y * y + (xi * d1 + z) * (xi * d1 + z));
      const Scalar& deno = alpha * d2 + (Scalar(1) - alpha) * (xi * d1 + z);

      res << fx * x / deno + cx, fy * y / deno + cy;
    }
    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(xi);
    UNUSED(alpha);
    UNUSED(x);
    UNUSED(y);
    UNUSED(z);

    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& xi = param[4];
    const Scalar& alpha = param[5];

    Vec3 res, res_temp, res_temp2;

    // TODO SHEET 2: implement camera model
    const Scalar& u = p[0];
    const Scalar& v = p[1];
    const Scalar& mx = (u - cx) / fx;
    const Scalar& my = (v - cy) / fy;

    const Scalar& r_square = mx * mx + my * my;
    const Scalar& mz =
        (Scalar(1) - alpha * alpha * r_square) /
        (alpha * sqrt(Scalar(1) - (Scalar(2) * alpha - Scalar(1)) * r_square) +
         (Scalar(1) - alpha));

    const Scalar& coef =
        (mz * xi + sqrt(mz * mz + (Scalar(1) - xi * xi) * r_square)) /
        (mz * mz + r_square);
    res_temp << mx, my, mz;
    res_temp2 << Scalar(0), Scalar(0), xi;

    res = coef * res_temp - res_temp2;
    UNUSED(p);
    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(xi);
    UNUSED(alpha);
    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar = double>
class KannalaBrandt4Camera : public AbstractCamera<Scalar> {
 public:
  static constexpr int N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vec4;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  KannalaBrandt4Camera() = default;
  KannalaBrandt4Camera(const VecN& p) : param(p) {}

  static KannalaBrandt4Camera getTestProjections() {
    VecN vec1;
    vec1 << 379.045, 379.008, 505.512, 509.969, 0.00693023, -0.0013828,
        -0.000272596, -0.000452646;
    KannalaBrandt4Camera res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "kb4"; }
  std::string name() const { return getName(); }

  inline Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    // TODO SHEET 2: implement camera model

    const Scalar& r = (p.head(2)).norm();
    const Scalar& theta = atan2(r, z);
    // std::cout<<"theta:"<< theta<<std::endl; 1.23
    const Scalar& theta_3 = theta * theta * theta;
    const Scalar& d_theta =
        theta + k1 * theta_3 + k2 * theta_3 * theta * theta +
        k3 * theta_3 * theta_3 * theta + k4 * theta_3 * theta_3 * theta_3;

    if (r != Scalar(0)) {
      res << fx * d_theta * x / r + cx, fy * d_theta * y / r + cy;

    } else {
      res.setZero();
    }

    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(k1);
    UNUSED(k2);
    UNUSED(k3);
    UNUSED(k4);
    UNUSED(x);
    UNUSED(y);
    UNUSED(z);

    return res;
  }

  Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    Vec3 res;

    // TODO SHEET 2: implement camera model
    const Scalar& u = p[0];
    const Scalar& v = p[1];

    Scalar mx = Scalar(0);
    Scalar my = Scalar(0);
    if (fx != Scalar(0) && u != Scalar(0)) {
      mx = (u - cx) / fx;
    }
    if (fy != Scalar(0) && v != Scalar(0)) {
      my = (v - cy) / fy;
    }

    const Scalar& r_u = sqrt(mx * mx + my * my);
    Scalar theta = Scalar(1.23);  // initial value from test data

    Scalar fun_theta = Scalar(0);

    Scalar fun_theta_deriv = Scalar(0);
    Scalar theta_new = Scalar(0);

    // theta_star is the solution of d_theta= r_u , use Newton's method
    // Scalar& theta=Scalar(1.23);// initial value from the test data
    int i = 0;
    while (i < 5) {
      Scalar theta_3 = theta * theta * theta;
      fun_theta_deriv = Scalar(1) + k1 * Scalar(3) * theta * theta +
                        k2 * Scalar(5) * theta_3 * theta +
                        k3 * Scalar(7) * theta_3 * theta_3 +
                        k4 * Scalar(9) * theta_3 * theta_3 * theta_3 / theta;
      fun_theta = theta + k1 * theta_3 + k2 * theta_3 * theta * theta +
                  k3 * theta_3 * theta_3 * theta +
                  k4 * theta_3 * theta_3 * theta_3 - r_u;

      if (fun_theta_deriv != Scalar(0)) {
        theta_new = theta - fun_theta / fun_theta_deriv;
      }
      // std::cout << "theta change:" << theta - theta_new << std::endl;

      if (abs(theta - theta_new) < Scalar(1e-16)) {
        // std::cout << "theta change inside:" << theta - theta_new <<
        // std::endl; p_normalized -0.666 666 666 666 666 63
        // -0.66666666666666663 0.33333333333333331 p_uproj -0.666 666 647 809
        // 945 56 -0.66666664780994556 0.33333340876020839 p_normalized 0 0 1
        // p_uproj -nan -nan -nan
        break;
      }
      theta = theta_new;
      i += 1;
    }

    // std::cout<<"theta change:"<< theta-Scalar(1.23)<<std::endl;

    if (r_u != Scalar(0)) {
      res << sin(theta) * mx / r_u, sin(theta) * my / r_u, cos(theta);
    } else {
      res << Scalar(0), Scalar(0), cos(theta);
    }

    UNUSED(p);
    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(k1);
    UNUSED(k2);
    UNUSED(k3);
    UNUSED(k4);

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar>
class AbstractCamera {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  virtual ~AbstractCamera() = default;

  virtual Scalar* data() = 0;

  virtual const Scalar* data() const = 0;

  virtual Vec2 project(const Vec3& p) const = 0;

  virtual Vec3 unproject(const Vec2& p) const = 0;

  virtual std::string name() const = 0;

  virtual const VecN& getParam() const = 0;

  inline int width() const { return width_; }
  inline int& width() { return width_; }
  inline int height() const { return height_; }
  inline int& height() { return height_; }

  static std::shared_ptr<AbstractCamera> from_data(const std::string& name,
                                                   const Scalar* sIntr) {
    if (name == DoubleSphereCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new DoubleSphereCamera<Scalar>(intr));
    } else if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(new PinholeCamera<Scalar>(intr));
    } else if (name == KannalaBrandt4Camera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new KannalaBrandt4Camera<Scalar>(intr));
    } else if (name == ExtendedUnifiedCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new ExtendedUnifiedCamera<Scalar>(intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }

  // Loading from double sphere initialization
  static std::shared_ptr<AbstractCamera> initialize(const std::string& name,
                                                    const Scalar* sIntr) {
    Eigen::Matrix<Scalar, 8, 1> init_intr;

    if (name == DoubleSphereCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;

      return std::shared_ptr<AbstractCamera>(
          new DoubleSphereCamera<Scalar>(init_intr));
    } else if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new PinholeCamera<Scalar>(init_intr));
    } else if (name == KannalaBrandt4Camera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new KannalaBrandt4Camera<Scalar>(init_intr));
    } else if (name == ExtendedUnifiedCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();
      init_intr[4] = 0.5;
      init_intr[5] = 1;

      return std::shared_ptr<AbstractCamera>(
          new ExtendedUnifiedCamera<Scalar>(init_intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }

 private:
  // image dimensions
  int width_ = 0;
  int height_ = 0;
};

}  // namespace visnav
