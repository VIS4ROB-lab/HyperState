/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <numeric>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "hyper/motion/interpolators/temporal/basis.hpp"

namespace hyper::state::tests {

using Scalar = double;
using Interpolator = BasisInterpolator<Scalar, Eigen::Dynamic>;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

using Index = Eigen::Index;
using IndexMatrix = std::map<Index, Matrix>;

constexpr auto kNumericTolerance = 1e-7;

TEST(BasisInterpolatorTests, Theory) {
  IndexMatrix theory;

  Matrix M1 = Matrix::Identity(1, 1);
  theory[1] = std::move(M1);

  Matrix M2 = Matrix::Identity(2, 2);
  theory[2] = std::move(M2);

  Matrix M3 = Matrix::Zero(3, 3);
  M3(0, 0) = Scalar{1};
  M3(1, 0) = Scalar{0.5};
  M3(1, 1) = Scalar{1};
  M3(1, 2) = Scalar{-0.5};
  M3(2, 2) = Scalar{0.5};
  theory[3] = std::move(M3);

  Matrix M4 = Matrix::Zero(4, 4);
  M4(0, 0) = Scalar{1};
  M4(1, 0) = Scalar{5.0 / 6.0};
  M4(2, 0) = Scalar{1.0 / 6.0};
  M4(1, 1) = Scalar{0.5};
  M4(2, 1) = Scalar{0.5};
  M4(1, 2) = Scalar{-0.5};
  M4(2, 2) = Scalar{0.5};
  M4(1, 3) = Scalar{1.0 / 6.0};
  M4(2, 3) = Scalar{-2.0 / 6.0};
  M4(3, 3) = Scalar{1.0 / 6.0};
  theory[4] = std::move(M4);

  for (const auto& [k, M] : theory) {
    EXPECT_TRUE(Interpolator::Mixing(k).isApprox(M, kNumericTolerance));
  }
}

TEST(BasisInterpolatorTests, Duality) {
  constexpr auto kMaxDegree = 5;
  for (Index i = 0; i < kMaxDegree; ++i) {
    Interpolator interpolator;
    interpolator.setOrder(i + 1);
    interpolator.setNonUniform();
    const auto layout = interpolator.layout();

    using Times = std::vector<Scalar>;
    Times times(layout.outer_input_size);
    std::iota(times.begin(), times.end(), 1 - layout.left_input_margin);

    const auto M0 = Interpolator::Mixing(i + 1);
    const auto M1 = interpolator.mixing(times);
    EXPECT_TRUE(M0.isApprox(M1, kNumericTolerance));
  }
}

}  // namespace hyper::state::tests
