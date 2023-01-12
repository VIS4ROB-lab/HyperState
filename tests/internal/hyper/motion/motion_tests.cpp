/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <fstream>

#include <gtest/gtest.h>

#include "hyper/variables/forward.hpp"

#include "hyper/motion/continuous.hpp"
#include "hyper/motion/interpolators/spatial/cartesian.hpp"
#include "hyper/motion/interpolators/spatial/se3.hpp"
#include "hyper/motion/interpolators/temporal/basis.hpp"
#include "hyper/variables/adapters.hpp"
#include "hyper/variables/groups/se3.hpp"

namespace hyper::tests {

using CartesianMotionTestTypes = ::testing::Types<std::tuple<BasisInterpolator<Scalar, Eigen::Dynamic>, Position<Scalar>>>;

template <typename TArgs>
class CartesianMotionTests : public testing::Test {
 public:
  static constexpr auto kDegree = 3;
  static constexpr auto kNumIterations = 20;
  static constexpr auto kNumericIncrement = 1e-8;
  static constexpr auto kNumericTolerance = 1e-6;

  using Interpolator = typename std::tuple_element<0, TArgs>::type;
  using Value = typename std::tuple_element<1, TArgs>::type;
  using StampedValue = Stamped<Value>;

  using Policy = SpatialInterpolator<Value>;
  using Tangent = typename Policy::Tangent;
  using Motion = ContinuousMotion<Value>;

  /// Set up.
  auto SetUp() -> void final {
    motion_ = ContinuousMotion<Value>{&interpolator_};
    interpolator_.setOrder(kDegree + 1);
  }

  /// Sets a random motion.
  auto setRandomMotion() -> void {
    const auto min_num_variables = motion_.interpolator()->layout().outer_input_size;
    for (auto i = Index{0}; i < min_num_variables + Eigen::internal::random<Index>(10, 20); ++i) {
      StampedValue stamped_value;
      stamped_value.stamp() = 0.25 * i;
      stamped_value.variable() = Value::Random();
      motion_.elements().insert(stamped_value);
    }
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(const Index degree = kDegree) -> bool {
    const auto stamp = motion_.range().sample();
    const auto derivative = static_cast<MotionDerivative>(degree);
    const auto result = motion_.evaluate(stamp, derivative, false);
    const auto d_result = motion_.evaluate(stamp + kNumericIncrement, derivative, false);

    for (auto i = 1; i <= degree; ++i) {
      const auto dx = ((d_result.derivative(i - 1) - result.derivative(i - 1)) / kNumericIncrement).eval();
      if (!dx.isApprox(result.derivative(i), kNumericTolerance)) return false;
    }

    return true;
  }

  /// Checks the Jacobians.
  /// \param degree Maximum derivative degree.
  /// \return True if numeric and analytic Jacobians are close.
  auto checkJacobians(const Index degree = kDegree) -> bool {
    for (Index i = 0; i <= degree; ++i) {
      // Evaluate analytic Jacobian.
      const auto stamp = motion_.range().sample();
      const auto derivative = static_cast<MotionDerivative>(i);
      const auto result = motion_.evaluate(stamp, derivative, true);

      // Retrieve inputs.
      const auto inputs = motion_.pointers(stamp);

      // Allocate Jacobian.
      JacobianX<Scalar> Jn_i;
      const auto num_inputs = static_cast<Index>(inputs.size());
      Jn_i.setZero(Value::kNumParameters, num_inputs * StampedValue::kNumParameters);

      // Evaluate Jacobian.
      for (Index j = 0; j < num_inputs; ++j) {
        auto input_j = Eigen::Map<StampedValue>{inputs[j]->asVector().data()};

        for (Index k = 0; k < input_j.size() - 1; ++k) {
          const StampedValue tmp = input_j;
          const Tangent tau = kNumericIncrement * Tangent::Unit(k);
          input_j.variable() += tau;

          const auto d_result = motion_.evaluate(stamp, derivative, false);
          Jn_i.col(j * StampedValue::kNumParameters + k) = (d_result.derivative(i) - result.derivative(i)).transpose() / kNumericIncrement;

          input_j = tmp;
        }
      }

      // Compare Jacobians.
      if (!Jn_i.isApprox(result.jacobian(i), kNumericTolerance)) return false;
    }

    return true;
  }

 private:
  Motion motion_;
  Interpolator interpolator_;
};

TYPED_TEST_SUITE_P(CartesianMotionTests);

TYPED_TEST_P(CartesianMotionTests, Derivatives) {
  this->setRandomMotion();
  for (auto i = 0; i < TestFixture::kNumIterations; ++i) {
    EXPECT_TRUE(this->checkDerivatives());
  }
}

TYPED_TEST_P(CartesianMotionTests, Jacobians) {
  this->setRandomMotion();
  for (auto i = 0; i < TestFixture::kNumIterations; ++i) {
    EXPECT_TRUE(this->checkJacobians());
  }
}

using ManifoldMotionTestTypes = ::testing::Types<std::tuple<BasisInterpolator<Scalar, Eigen::Dynamic>, SE3<Scalar>>>;

template <typename TArgs>
class ManifoldMotionTests : public testing::Test {
 public:
  static constexpr auto kDegree = 3;
  static constexpr auto kNumIterations = 20;
  static constexpr auto kNumericIncrement = 1e-8;
  static constexpr auto kNumericTolerance = 1e-6;

  using Interpolator = typename std::tuple_element<0, TArgs>::type;
  using Value = typename std::tuple_element<1, TArgs>::type;
  using StampedValue = Stamped<Value>;

  using Policy = SpatialInterpolator<Value>;
  using Tangent = typename Policy::Tangent;
  using Motion = ContinuousMotion<Value>;

  /// Sets a random motion.
  auto setRandomMotion() -> void {
    const auto min_num_variables = motion_.interpolator()->layout().outer_input_size;
    for (auto i = Index{0}; i < min_num_variables + Eigen::internal::random<Index>(10, 20); ++i) {
      StampedValue stamped_value;
      stamped_value.stamp() = 0.25 * i;
      stamped_value.variable() = Value::Random();
      motion_.elements().insert(stamped_value);
    }
  }

  /// Set up.
  auto SetUp() -> void final {
    motion_ = ContinuousMotion<Value>{&interpolator_};
    interpolator_.setOrder(kDegree + 1);
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(const Index degree = kDegree) -> bool {
    const auto stamp = motion_.range().sample();
    const auto derivative = static_cast<MotionDerivative>(degree);
    const auto result = motion_.evaluate(stamp, derivative, false);
    const auto d_result = motion_.evaluate(stamp + kNumericIncrement, derivative, false);

    const auto value = result.value();
    const auto d_value = d_result.value();

    for (Index i = 1; i <= degree; ++i) {
      Tangent dx;

      if (i == 1) {
        SU2<Scalar> d_su2;
        Algebra<SU2<Scalar>> d_algebra;
        d_su2 = SU2<Scalar>{(d_value.rotation().coeffs() - value.rotation().coeffs()) / kNumericIncrement};
        d_algebra = value.rotation().groupInverse().groupPlus(d_su2).coeffs();
        dx.angular() = d_algebra.toTangent();
        dx.linear() = (d_value.translation() - value.translation()) / kNumericIncrement;
      } else {
        dx = (d_result.derivative(i - 1) - result.derivative(i - 1)) / kNumericIncrement;
      }

      if (!dx.isApprox(result.derivative(i), kNumericTolerance)) return false;
    }

    return true;
  }

  /// Checks the Jacobians.
  /// \param degree Maximum derivative degree.
  /// \return True if numeric and analytic Jacobians are close.
  auto checkJacobians(const Index degree = kDegree) -> bool {
    for (Index i = 0; i <= degree; ++i) {
      const auto stamp = motion_.range().sample();
      const auto derivative = static_cast<MotionDerivative>(degree);
      const auto result = motion_.evaluate(stamp, derivative, true);

      // Retrieve inputs.
      const auto inputs = motion_.pointers(stamp);

      // Allocate Jacobian.
      JacobianX<Scalar> Jn_i;
      const auto num_inputs = static_cast<Index>(inputs.size());
      Jn_i.setZero(Tangent::kNumParameters, num_inputs * StampedValue::kNumParameters);

      // Evaluate Jacobian.
      for (Index j = 0; j < num_inputs; ++j) {
        auto input_j = Eigen::Map<StampedValue>{inputs[j]->asVector().data()};

        for (Index k = 0; k < Tangent::kNumParameters; ++k) {
          const StampedValue tmp = input_j;
          const Tangent tau = kNumericIncrement * Tangent::Unit(k);
          input_j.variable().rotation() *= tau.angular().toManifold();
          input_j.variable().translation() += tau.linear();

          const auto d_result = motion_.evaluate(stamp, derivative, false);

          if (i == 0) {
            const auto value = result.value();
            const auto d_value = d_result.value();
            Jn_i.col(j * StampedValue::kNumParameters + k).template head<3>() = (value.rotation().groupInverse().groupPlus(d_value.rotation())).toTangent() / kNumericIncrement;
            Jn_i.col(j * StampedValue::kNumParameters + k).template tail<3>() = (d_value.translation() - value.translation()) / kNumericIncrement;
          } else {
            Jn_i.col(j * StampedValue::kNumParameters + k) = (d_result.derivative(i) - result.derivative(i)) / kNumericIncrement;
          }

          input_j = tmp;
        }

        Jn_i.template middleCols<StampedValue::kNumParameters - 1>(j * StampedValue::kNumParameters) = Jn_i.template middleCols<Tangent::kNumParameters>(j * StampedValue::kNumParameters) * SE3JacobianAdapter(inputs[j]->asVector().data());
      }

      // Compare Jacobians.
      if (!Jn_i.isApprox(result.jacobian(i), kNumericTolerance)) return false;
    }

    return true;
  }

 private:
  Motion motion_;
  Interpolator interpolator_;
};

TYPED_TEST_SUITE_P(ManifoldMotionTests);

TYPED_TEST_P(ManifoldMotionTests, Derivatives) {
  this->setRandomMotion();
  for (auto i = 0; i < TestFixture::kNumIterations; ++i) {
    EXPECT_TRUE(this->checkDerivatives(2));
  }
}

TYPED_TEST_P(ManifoldMotionTests, Jacobians) {
  this->setRandomMotion();
  for (auto i = 0; i < TestFixture::kNumIterations; ++i) {
    EXPECT_TRUE(this->checkJacobians(2));
  }
}

REGISTER_TYPED_TEST_SUITE_P(CartesianMotionTests, Derivatives, Jacobians);
INSTANTIATE_TYPED_TEST_SUITE_P(HyperTests, CartesianMotionTests, CartesianMotionTestTypes);

REGISTER_TYPED_TEST_SUITE_P(ManifoldMotionTests, Derivatives, Jacobians);
INSTANTIATE_TYPED_TEST_SUITE_P(HyperTests, ManifoldMotionTests, ManifoldMotionTestTypes);

} // namespace hyper::tests
