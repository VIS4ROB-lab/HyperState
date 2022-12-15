/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <fstream>

#include <gtest/gtest.h>

#include "hyper/variables/forward.hpp"

#include "hyper/state/abstract.hpp"
#include "hyper/state/interpolators/basis.hpp"
#include "hyper/state/policies/cartesian.hpp"

#include "hyper/state/policies/se3.hpp"
#include "hyper/variables/adapters.hpp"
#include "hyper/variables/groups/se3.hpp"

namespace hyper::tests {

using CartesianStateTestTypes = ::testing::Types<std::tuple<BasisInterpolator<Scalar, Eigen::Dynamic>, Position<Scalar>>>;

template <typename TArgs>
class CartesianStateTests : public testing::Test {
 public:
  static constexpr auto kDegree = 3;
  static constexpr auto kNumIterations = 20;
  static constexpr auto kNumericIncrement = 1e-8;
  static constexpr auto kNumericTolerance = 1e-6;

  using Interpolator = typename std::tuple_element<0, TArgs>::type;
  using Value = typename std::tuple_element<1, TArgs>::type;
  using StampedValue = Stamped<Value>;

  using Jacobian = StateResult::Jacobian;
  using Jacobians = StateResult::Jacobians;
  using Policy = CartesianPolicy<StampedValue>;

  using Input = typename Policy::Input;
  using Derivative = typename Policy::Derivative;

  /// Set up.
  auto SetUp() -> void final {
    auto interpolator = std::make_unique<Interpolator>();
    interpolator->setOrder(kDegree + 1);
    auto policy = std::make_unique<Policy>();
    state_ = AbstractState{std::move(interpolator), std::move(policy)};
  }

  /// Sets a random state.
  auto setRandomState() -> void {
    const auto min_num_variables = state_.interpolator()->layout().outer_input_size;
    for (auto i = Index{0}; i < min_num_variables + Eigen::internal::random<Index>(10, 20); ++i) {
      auto input = std::make_unique<Input>();
      input->stamp() = i;
      input->variable() = Value::Random();
      state_.elements().insert(std::move(input));
    }
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(const Index degree = kDegree) -> bool {
    const auto stamp = state_.range().sample();
    const auto query = StateQuery{stamp, degree, false};
    auto output = state_.evaluate(query);

    const auto d_query = StateQuery{stamp + kNumericIncrement, degree, false};
    auto d_output = state_.evaluate(d_query);

    for (auto i = 1; i <= degree; ++i) {
      const auto derivative = ((d_output.derivatives.at(i - 1) - output.derivatives.at(i - 1)) / kNumericIncrement).eval();
      if (!derivative.isApprox(output.derivatives.at(i), kNumericTolerance)) return false;
    }

    return true;
  }

  /// Checks the Jacobians.
  /// \param degree Maximum derivative degree.
  /// \return True if numeric and analytic Jacobians are close.
  auto checkJacobians(const Index degree = kDegree) -> bool {
    for (Index i = 0; i <= degree; ++i) {
      // Evaluate analytic Jacobian.
      const auto stamp = state_.range().sample();
      const auto query = StateQuery{stamp, i, true};
      auto output = state_.evaluate(query);

      // Retrieve inputs.
      const auto inputs = state_.parameters(stamp);

      // Allocate Jacobian.
      Jacobian Jn_i;
      const auto num_inputs = static_cast<Index>(inputs.size());
      Jn_i.setZero(Value::kNumParameters, num_inputs * Input::kNumParameters);

      // Evaluate Jacobian.
      for (Index j = 0; j < num_inputs; ++j) {
        auto input_j = Eigen::Map<Input>{inputs[j]->asVector().data()};

        for (Index k = 0; k < input_j.size() - 1; ++k) {
          const Input tmp = input_j;
          const Derivative tau = kNumericIncrement * Derivative::Unit(k);
          input_j.variable() += tau;

          const auto d_query = StateQuery{stamp, i, false};
          const auto d_output = state_.evaluate(d_query);
          Jn_i.col(j * Input::kNumParameters + k) = (d_output.derivatives.at(i) - output.derivatives.at(i)).transpose() / kNumericIncrement;

          input_j = tmp;
        }
      }

      // Compare Jacobians.
      if (!Jn_i.isApprox(output.jacobians.at(i), kNumericTolerance)) return false;
    }

    return true;
  }

 private:
  AbstractState state_;
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

using ManifoldStateTestTypes = ::testing::Types<std::tuple<BasisInterpolator<Scalar, Eigen::Dynamic>, SE3<Scalar>>>;

template <typename TArgs>
class ManifoldStateTests : public testing::Test {
 public:
  static constexpr auto kDegree = 3;
  static constexpr auto kNumIterations = 20;
  static constexpr auto kNumericIncrement = 1e-8;
  static constexpr auto kNumericTolerance = 1e-6;

  using Interpolator = typename std::tuple_element<0, TArgs>::type;
  using Value = typename std::tuple_element<1, TArgs>::type;
  using StampedValue = Stamped<Value>;

  using Jacobian = StateResult::Jacobian;
  using Jacobians = StateResult::Jacobians;
  using Policy = ManifoldPolicy<StampedValue>;

  using Input = typename Policy::Input;
  using Derivative = typename Policy::Derivative;

  /// Sets a random state.
  auto setRandomState() -> void {
    const auto min_num_variables = state_.interpolator()->layout().outer_input_size;
    for (auto i = Index{0}; i < min_num_variables + Eigen::internal::random<Index>(10, 20); ++i) {
      auto input = std::make_unique<Input>();
      input->stamp() = i;
      input->variable() = Value::Random();
      state_.elements().insert(std::move(input));
    }
  }

  /// Set up.
  auto SetUp() -> void final {
    auto interpolator = std::make_unique<Interpolator>();
    interpolator->setOrder(kDegree + 1);
    auto policy = std::make_unique<Policy>();
    state_ = AbstractState{std::move(interpolator), std::move(policy)};
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(const Index degree = kDegree) -> bool {
    const auto stamp = state_.range().sample();
    const auto query = StateQuery{stamp, degree, false};
    auto output = state_.evaluate(query);

    const auto d_query = StateQuery{stamp + kNumericIncrement, degree, false};
    auto d_output = state_.evaluate(d_query);

    const auto value = output.template derivativeAs<SE3<Scalar>>(0);
    const auto d_value = d_output.template derivativeAs<SE3<Scalar>>(0);

    for (Index i = 1; i <= degree; ++i) {
      Derivative derivative;

      if (i == 1) {
        SU2<Scalar> d_su2;
        Algebra<SU2<Scalar>> d_algebra;
        d_su2 = SU2<Scalar>{(d_value.rotation().coeffs() - value.rotation().coeffs()) / kNumericIncrement};
        d_algebra = value.rotation().groupInverse().groupPlus(d_su2).coeffs();
        derivative.angular() = d_algebra.toTangent();
        derivative.linear() = (d_value.translation() - value.translation()) / kNumericIncrement;
      } else {
        derivative = (d_output.derivatives.at(i - 1) - output.derivatives.at(i - 1)) / kNumericIncrement;
      }

      if (!derivative.isApprox(output.derivatives[i], kNumericTolerance)) return false;
    }

    return true;
  }

  /// Checks the Jacobians.
  /// \param degree Maximum derivative degree.
  /// \return True if numeric and analytic Jacobians are close.
  auto checkJacobians(const Index degree = kDegree) -> bool {
    for (Index i = 0; i <= degree; ++i) {
      const auto stamp = state_.range().sample();
      const auto query = StateQuery{stamp, degree, true};
      auto output = state_.evaluate(query);

      // Retrieve inputs.
      const auto inputs = state_.parameters(stamp);

      // Allocate Jacobian.
      Jacobian Jn_i;
      const auto num_inputs = static_cast<Index>(inputs.size());
      Jn_i.setZero(Derivative::kNumParameters, num_inputs * Input::kNumParameters);

      // Evaluate Jacobian.
      for (Index j = 0; j < num_inputs; ++j) {
        auto input_j = Eigen::Map<Input>{inputs[j]->asVector().data()};

        for (Index k = 0; k < Derivative::kNumParameters; ++k) {
          const Input tmp = input_j;
          const Derivative tau = kNumericIncrement * Derivative::Unit(k);
          input_j.variable().rotation() *= tau.angular().toManifold();
          input_j.variable().translation() += tau.linear();

          const auto d_query = StateQuery{stamp, degree, false};
          const auto d_output = state_.evaluate(d_query);

          if (i == 0) {
            const auto value = output.template derivativeAs<SE3<Scalar>>(0);
            const auto d_value = d_output.template derivativeAs<SE3<Scalar>>(0);
            Jn_i.col(j * Input::kNumParameters + k).template head<3>() = (value.rotation().groupInverse().groupPlus(d_value.rotation())).toTangent() / kNumericIncrement;
            Jn_i.col(j * Input::kNumParameters + k).template tail<3>() = (d_value.translation() - value.translation()) / kNumericIncrement;
          } else {
            Jn_i.col(j * Input::kNumParameters + k) = (d_output.derivatives.at(i) - output.derivatives.at(i)) / kNumericIncrement;
          }

          input_j = tmp;
        }

        Jn_i.template middleCols<Input::kNumParameters - 1>(j * Input::kNumParameters) = Jn_i.template middleCols<Derivative::kNumParameters>(j * Input::kNumParameters) * SE3JacobianAdapter(inputs[j]->asVector().data());
      }

      // Compare Jacobians.
      if (!Jn_i.isApprox(output.jacobians.at(i), kNumericTolerance)) return false;
    }

    return true;
  }

 private:
  AbstractState state_;
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
    EXPECT_TRUE(this->checkJacobians(1));
  }
}

REGISTER_TYPED_TEST_SUITE_P(CartesianStateTests, Derivatives, Jacobians);
INSTANTIATE_TYPED_TEST_SUITE_P(HyperTests, CartesianStateTests, CartesianStateTestTypes);

REGISTER_TYPED_TEST_SUITE_P(ManifoldStateTests, Derivatives, Jacobians);
INSTANTIATE_TYPED_TEST_SUITE_P(HyperTests, ManifoldStateTests, ManifoldStateTestTypes);

} // namespace hyper::tests
