/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <fstream>

#include <gtest/gtest.h>

#include "hyper/variables/forward.hpp"

#include "hyper/state/continuous.hpp"
#include "hyper/state/interpolators/interpolators.hpp"
#include "hyper/variables/groups/adapters.hpp"

namespace hyper::state::tests {

using CartesianStateTestTypes = ::testing::Types<std::tuple<BasisInterpolator<double, Eigen::Dynamic>, variables::Position<double>>>;

template <typename TArgs>
class CartesianStateTests : public testing::Test {
 public:
  // Constants.
  static constexpr auto kDegree = 3;
  static constexpr auto kNumIterations = 20;
  static constexpr auto kNumericIncrement = 1e-8;
  static constexpr auto kNumericTolerance = 1e-6;

  // Definitions.
  using Index = Eigen::Index;
  using Interpolator = typename std::tuple_element<0, TArgs>::type;
  using Manifold = typename std::tuple_element<1, TArgs>::type;

  using Scalar = typename Manifold::Scalar;
  using Tangent = variables::Tangent<Manifold>;
  using StampedManifold = variables::Stamped<Manifold>;
  using State = ContinuousState<Manifold>;

  using Jacobian = variables::JacobianX<Scalar>;

  /// Set up.
  auto SetUp() -> void final {
    state_ = ContinuousState<Manifold>{&interpolator_};
    interpolator_.setOrder(kDegree + 1);
  }

  /// Sets a random state.
  auto setRandomState() -> void {
    const auto min_num_variables = state_.interpolator()->layout().outer_input_size;
    for (auto i = Index{0}; i < min_num_variables + Eigen::internal::random<Index>(10, 20); ++i) {
      StampedManifold stamped_manifold;
      stamped_manifold.stamp() = 0.25 * i;
      stamped_manifold.variable() = Manifold::Random();
      state_.elements().insert(stamped_manifold);
    }
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(const Index degree = kDegree) -> bool {
    const auto time = state_.range().sample();
    const auto result = state_.evaluate(time, degree, false);
    const auto d_result = state_.evaluate(time + kNumericIncrement, degree, false);

    Tangent dx;
    for (Index i = 0; i < degree; ++i) {
      if (i == 0) {
        dx = (d_result.value() - result.value()) / kNumericIncrement;
      } else {
        dx = (d_result.derivative(i - 1) - result.derivative(i - 1)) / kNumericIncrement;
      }
      if (!dx.isApprox(result.derivative(i), kNumericTolerance))
        return false;
    }

    return true;
  }

  /// Checks the Jacobians.
  /// \param degree Maximum derivative degree.
  /// \return True if numeric and analytic Jacobians are close.
  auto checkJacobians(const Index degree = kDegree) -> bool {
    for (Index i = 0; i <= degree; ++i) {
      // Evaluate analytic Jacobian.
      const auto time = state_.range().sample();
      const auto result = state_.evaluate(time, i, true);

      // Retrieve inputs.
      const auto inputs = state_.variables(time);

      // Allocate Jacobian.
      Jacobian Jn_i;
      const auto num_inputs = static_cast<Index>(inputs.size());
      Jn_i.setZero(Manifold::kNumParameters, num_inputs * StampedManifold::kNumParameters);

      // Evaluate Jacobian.
      for (Index j = 0; j < num_inputs; ++j) {
        auto input_j = Eigen::Map<StampedManifold>{inputs[j]->data()};

        for (Index k = 0; k < input_j.size() - 1; ++k) {
          const StampedManifold tmp = input_j;
          const Tangent tau = kNumericIncrement * Tangent::Unit(k);
          input_j.variable() += tau;

          const auto d_result = state_.evaluate(time, i, false);

          if (i == 0) {
            Jn_i.col(j * StampedManifold::kNumParameters + k) = (d_result.value() - result.value()).transpose() / kNumericIncrement;
          } else {
            Jn_i.col(j * StampedManifold::kNumParameters + k) = (d_result.derivative(i - 1) - result.derivative(i - 1)).transpose() / kNumericIncrement;
          }

          input_j = tmp;
        }
      }

      // Compare Jacobians.
      if (!Jn_i.isApprox(result.jacobian(i), kNumericTolerance))
        return false;
    }

    return true;
  }

 private:
  State state_;
  Interpolator interpolator_;
};

TYPED_TEST_SUITE_P(CartesianStateTests);

TYPED_TEST_P(CartesianStateTests, Derivatives) {
  this->setRandomState();
  for (auto i = 0; i < TestFixture::kNumIterations; ++i) {
    EXPECT_TRUE(this->checkDerivatives());
  }
}

TYPED_TEST_P(CartesianStateTests, Jacobians) {
  this->setRandomState();
  for (auto i = 0; i < TestFixture::kNumIterations; ++i) {
    EXPECT_TRUE(this->checkJacobians());
  }
}

using ManifoldStateTestTypes = ::testing::Types<std::tuple<BasisInterpolator<double, Eigen::Dynamic>, variables::SE3<double>>>;

template <typename TArgs>
class ManifoldStateTests : public testing::Test {
 public:
  // Constants.
  static constexpr auto kDegree = 3;
  static constexpr auto kNumIterations = 20;
  static constexpr auto kNumericIncrement = 1e-8;
  static constexpr auto kNumericTolerance = 1e-6;

  // Definitions.
  using Index = Eigen::Index;
  using Interpolator = typename std::tuple_element<0, TArgs>::type;
  using Manifold = typename std::tuple_element<1, TArgs>::type;

  using Scalar = typename Manifold::Scalar;
  using Tangent = variables::Tangent<Manifold>;
  using StampedManifold = variables::Stamped<Manifold>;
  using State = ContinuousState<Manifold>;

  using SU2 = variables::SU2<Scalar>;
  using SU2Algebra = variables::Algebra<SU2>;
  using Jacobian = variables::JacobianX<Scalar>;

  /// Sets a random state.
  auto setRandomState() -> void {
    const auto min_num_variables = state_.interpolator()->layout().outer_input_size;
    for (auto i = Index{0}; i < min_num_variables + Eigen::internal::random<Index>(10, 20); ++i) {
      StampedManifold stamped_manifold;
      stamped_manifold.stamp() = 0.25 * i;
      stamped_manifold.variable() = Manifold::Random();
      state_.elements().insert(stamped_manifold);
    }
  }

  /// Set up.
  auto SetUp() -> void final {
    state_ = ContinuousState<Manifold>{&interpolator_};
    interpolator_.setOrder(kDegree + 1);
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(const Index degree = kDegree) -> bool {
    const auto time = state_.range().sample();
    const auto result = state_.evaluate(time, degree, false);
    const auto d_result = state_.evaluate(time + kNumericIncrement, degree, false);

    const auto& value = result.value();
    const auto& d_value = d_result.value();

    Tangent dx;
    for (Index i = 0; i < degree; ++i) {
      if (i == 0) {
        SU2 d_su2;
        SU2Algebra d_algebra;
        d_su2 = SU2{(d_value.rotation().coeffs() - value.rotation().coeffs()) / kNumericIncrement};
        d_algebra = value.rotation().groupInverse().groupPlus(d_su2).coeffs();
        dx.angular() = d_algebra.toTangent();
        dx.linear() = (d_value.translation() - value.translation()) / kNumericIncrement;
      } else {
        dx = (d_result.derivative(i - 1) - result.derivative(i - 1)) / kNumericIncrement;
      }

      if (!dx.isApprox(result.derivative(i), kNumericTolerance))
        return false;
    }

    return true;
  }

  /// Checks the Jacobians.
  /// \param degree Maximum derivative degree.
  /// \return True if numeric and analytic Jacobians are close.
  auto checkJacobians(const Index degree = kDegree) -> bool {
    for (Index i = 0; i <= degree; ++i) {
      const auto time = state_.range().sample();
      const auto result = state_.evaluate(time, degree, true);

      // Retrieve inputs.
      const auto inputs = state_.variables(time);

      // Allocate Jacobian.
      Jacobian Jn_i;
      const auto num_inputs = static_cast<Index>(inputs.size());
      Jn_i.setZero(Tangent::kNumParameters, num_inputs * StampedManifold::kNumParameters);

      // Evaluate Jacobian.
      for (Index j = 0; j < num_inputs; ++j) {
        auto input_j = Eigen::Map<StampedManifold>{inputs[j]->data()};

        for (Index k = 0; k < Tangent::kNumParameters; ++k) {
          const StampedManifold tmp = input_j;
          const Tangent tau = kNumericIncrement * Tangent::Unit(k);
          input_j.variable().rotation() *= tau.angular().toManifold();
          input_j.variable().translation() += tau.linear();

          const auto d_result = state_.evaluate(time, degree, false);

          if (i == 0) {
            const auto& value = result.value();
            const auto& d_value = d_result.value();
            Jn_i.col(j * StampedManifold::kNumParameters + k).template head<3>() = (value.rotation().groupInverse().groupPlus(d_value.rotation())).toTangent() / kNumericIncrement;
            Jn_i.col(j * StampedManifold::kNumParameters + k).template tail<3>() = (d_value.translation() - value.translation()) / kNumericIncrement;
          } else {
            Jn_i.col(j * StampedManifold::kNumParameters + k) = (d_result.derivative(i - 1) - result.derivative(i - 1)) / kNumericIncrement;
          }

          input_j = tmp;
        }

        Jn_i.template middleCols<StampedManifold::kNumParameters - 1>(j * StampedManifold::kNumParameters) =
            Jn_i.template middleCols<Tangent::kNumParameters>(j * StampedManifold::kNumParameters) * variables::JacobianAdapter<Manifold>(inputs[j]->data());
      }

      // Compare Jacobians.
      if (!Jn_i.isApprox(result.jacobian(i), kNumericTolerance))
        return false;
    }

    return true;
  }

 private:
  State state_;
  Interpolator interpolator_;
};

TYPED_TEST_SUITE_P(ManifoldStateTests);

TYPED_TEST_P(ManifoldStateTests, Derivatives) {
  this->setRandomState();
  for (auto i = 0; i < TestFixture::kNumIterations; ++i) {
    EXPECT_TRUE(this->checkDerivatives(2));
  }
}

TYPED_TEST_P(ManifoldStateTests, Jacobians) {
  this->setRandomState();
  for (auto i = 0; i < TestFixture::kNumIterations; ++i) {
    EXPECT_TRUE(this->checkJacobians(2));
  }
}

REGISTER_TYPED_TEST_SUITE_P(CartesianStateTests, Derivatives, Jacobians);
INSTANTIATE_TYPED_TEST_SUITE_P(HyperTests, CartesianStateTests, CartesianStateTestTypes);

REGISTER_TYPED_TEST_SUITE_P(ManifoldStateTests, Derivatives, Jacobians);
INSTANTIATE_TYPED_TEST_SUITE_P(HyperTests, ManifoldStateTests, ManifoldStateTestTypes);

}  // namespace hyper::state::tests
