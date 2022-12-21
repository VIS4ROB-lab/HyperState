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

  using Policy = SpatialInterpolator<StampedValue>;

  using Input = typename Policy::Input;
  using Tangent = typename Policy::Tangent;

  using Motion = ContinuousMotion<Value>;
  using Query = typename Motion::Query;

  /// Set up.
  auto SetUp() -> void final {
    auto interpolator = std::make_unique<Interpolator>(kDegree + 1);
    motion_ = ContinuousMotion<Value>{std::move(interpolator)};
  }

  /// Sets a random motion.
  auto setRandomMotion() -> void {
    const auto min_num_variables = motion_.interpolator()->layout().outer_input_size;
    for (auto i = Index{0}; i < min_num_variables + Eigen::internal::random<Index>(10, 20); ++i) {
      Input input;
      input.stamp() = 0.25 * i;
      input.variable() = Value::Random();
      motion_.elements().insert(input);
    }
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(const Index degree = kDegree) -> bool {
    const auto stamp = motion_.range().sample();
    const auto query = Query{stamp, static_cast<MotionDerivative>(degree), false};
    const auto result = motion_.evaluate(query);

    const auto d_query = Query{stamp + kNumericIncrement, static_cast<MotionDerivative>(degree), false};
    const auto d_result = motion_.evaluate(d_query);

    for (auto i = 1; i <= degree; ++i) {
      const auto derivative = ((d_result.derivatives.at(i - 1) - result.derivatives.at(i - 1)) / kNumericIncrement).eval();
      if (!derivative.isApprox(result.derivatives.at(i), kNumericTolerance)) return false;
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
      const auto query = Query{stamp, static_cast<MotionDerivative>(i), true};
      const auto result = motion_.evaluate(query);

      // Retrieve inputs.
      const auto inputs = motion_.pointers(stamp);

      // Allocate Jacobian.
      JacobianX<Scalar> Jn_i;
      const auto num_inputs = static_cast<Index>(inputs.size());
      Jn_i.setZero(Value::kNumParameters, num_inputs * Input::kNumParameters);

      // Evaluate Jacobian.
      for (Index j = 0; j < num_inputs; ++j) {
        auto input_j = Eigen::Map<Input>{inputs[j]->asVector().data()};

        for (Index k = 0; k < input_j.size() - 1; ++k) {
          const Input tmp = input_j;
          const Tangent tau = kNumericIncrement * Tangent::Unit(k);
          input_j.variable() += tau;

          const auto d_query = Query{stamp, static_cast<MotionDerivative>(i), false};
          const auto d_result = motion_.evaluate(d_query);
          Jn_i.col(j * Input::kNumParameters + k) = (d_result.derivatives.at(i) - result.derivatives.at(i)).transpose() / kNumericIncrement;

          input_j = tmp;
        }
      }

      // Compare Jacobians.
      if (!Jn_i.isApprox(result.jacobians.at(i), kNumericTolerance)) return false;
    }

    return true;
  }

 private:
  Motion motion_;
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

  using Policy = SpatialInterpolator<StampedValue>;

  using Input = typename Policy::Input;
  using Tangent = typename Policy::Tangent;

  using Motion = ContinuousMotion<Value>;
  using Query = typename Motion::Query;

  /// Sets a random motion.
  auto setRandomMotion() -> void {
    const auto min_num_variables = motion_.interpolator()->layout().outer_input_size;
    for (auto i = Index{0}; i < min_num_variables + Eigen::internal::random<Index>(10, 20); ++i) {
      Input input;
      input.stamp() = 0.25 * i;
      input.variable() = Value::Random();
      motion_.elements().insert(input);
    }
  }

  /// Set up.
  auto SetUp() -> void final {
    auto interpolator = std::make_unique<Interpolator>(kDegree + 1);
    motion_ = ContinuousMotion<Value>{std::move(interpolator)};
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(const Index degree = kDegree) -> bool {
    const auto stamp = motion_.range().sample();
    const auto query = Query{stamp, static_cast<MotionDerivative>(degree), false};
    const auto result = motion_.evaluate(query);

    const auto d_query = Query{stamp + kNumericIncrement, static_cast<MotionDerivative>(degree), false};
    const auto d_result = motion_.evaluate(d_query);

    const auto value = result.template derivativeAs<SE3<Scalar>>(0);
    const auto d_value = d_result.template derivativeAs<SE3<Scalar>>(0);

    for (Index i = 1; i <= degree; ++i) {
      Tangent derivative;

      if (i == 1) {
        SU2<Scalar> d_su2;
        Algebra<SU2<Scalar>> d_algebra;
        d_su2 = SU2<Scalar>{(d_value.rotation().coeffs() - value.rotation().coeffs()) / kNumericIncrement};
        d_algebra = value.rotation().groupInverse().groupPlus(d_su2).coeffs();
        derivative.angular() = d_algebra.toTangent();
        derivative.linear() = (d_value.translation() - value.translation()) / kNumericIncrement;
      } else {
        derivative = (d_result.derivatives.at(i - 1) - result.derivatives.at(i - 1)) / kNumericIncrement;
      }

      if (!derivative.isApprox(result.derivatives[i], kNumericTolerance)) return false;
    }

    return true;
  }

  /// Checks the Jacobians.
  /// \param degree Maximum derivative degree.
  /// \return True if numeric and analytic Jacobians are close.
  auto checkJacobians(const Index degree = kDegree) -> bool {
    for (Index i = 0; i <= degree; ++i) {
      const auto stamp = motion_.range().sample();
      const auto query = Query{stamp, static_cast<MotionDerivative>(degree), true};
      const auto result = motion_.evaluate(query);

      // Retrieve inputs.
      const auto inputs = motion_.pointers(stamp);

      // Allocate Jacobian.
      JacobianX<Scalar> Jn_i;
      const auto num_inputs = static_cast<Index>(inputs.size());
      Jn_i.setZero(Tangent::kNumParameters, num_inputs * Input::kNumParameters);

      // Evaluate Jacobian.
      for (Index j = 0; j < num_inputs; ++j) {
        auto input_j = Eigen::Map<Input>{inputs[j]->asVector().data()};

        for (Index k = 0; k < Tangent::kNumParameters; ++k) {
          const Input tmp = input_j;
          const Tangent tau = kNumericIncrement * Tangent::Unit(k);
          input_j.variable().rotation() *= tau.angular().toManifold();
          input_j.variable().translation() += tau.linear();

          const auto d_query = Query{stamp, static_cast<MotionDerivative>(degree), false};
          const auto d_result = motion_.evaluate(d_query);

          if (i == 0) {
            const auto value = result.template derivativeAs<SE3<Scalar>>(0);
            const auto d_value = d_result.template derivativeAs<SE3<Scalar>>(0);
            Jn_i.col(j * Input::kNumParameters + k).template head<3>() = (value.rotation().groupInverse().groupPlus(d_value.rotation())).toTangent() / kNumericIncrement;
            Jn_i.col(j * Input::kNumParameters + k).template tail<3>() = (d_value.translation() - value.translation()) / kNumericIncrement;
          } else {
            Jn_i.col(j * Input::kNumParameters + k) = (d_result.derivatives.at(i) - result.derivatives.at(i)) / kNumericIncrement;
          }

          input_j = tmp;
        }

        Jn_i.template middleCols<Input::kNumParameters - 1>(j * Input::kNumParameters) = Jn_i.template middleCols<Tangent::kNumParameters>(j * Input::kNumParameters) * SE3JacobianAdapter(inputs[j]->asVector().data());
      }

      // Compare Jacobians.
      if (!Jn_i.isApprox(result.jacobians.at(i), kNumericTolerance)) return false;
    }

    return true;
  }

 private:
  Motion motion_;
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
    EXPECT_TRUE(this->checkJacobians(1));
  }
}

REGISTER_TYPED_TEST_SUITE_P(CartesianMotionTests, Derivatives, Jacobians);
INSTANTIATE_TYPED_TEST_SUITE_P(HyperTests, CartesianMotionTests, CartesianMotionTestTypes);

REGISTER_TYPED_TEST_SUITE_P(ManifoldMotionTests, Derivatives, Jacobians);
INSTANTIATE_TYPED_TEST_SUITE_P(HyperTests, ManifoldMotionTests, ManifoldMotionTestTypes);

} // namespace hyper::tests
