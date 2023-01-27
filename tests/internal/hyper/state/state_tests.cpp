/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <fstream>

#include <gtest/gtest.h>

#include "hyper/variables/forward.hpp"

#include "hyper/state/continuous.hpp"
#include "hyper/state/interpolators/interpolators.hpp"
#include "hyper/variables/groups/adapters.hpp"

namespace hyper::state::tests {

using CartesianStateTestTypes = ::testing::Types<std::tuple<ContinuousState<variables::Position<double>>, BasisInterpolator<double, Eigen::Dynamic>>>;

template <typename TArgs>
class CartesianStateTests : public testing::Test {
 public:
  // Constants.
  static constexpr auto kDegree = 3;
  static constexpr auto kItr = 20;
  static constexpr auto kInc = 1e-8;
  static constexpr auto kTol = 1e-6;

  // Definitions.
  using State = typename std::tuple_element<0, TArgs>::type;
  using Interpolator = typename std::tuple_element<1, TArgs>::type;

  using Index = typename State::Index;
  using Scalar = typename State::Scalar;
  using Variable = typename State::Variable;
  using Output = typename State::Output;
  using StampedVariable = typename State::StampedVariable;

  using Tangent = variables::Tangent<Output>;
  using Jacobian = variables::JacobianX<Scalar>;

  /// Set up.
  auto SetUp() -> void final {
    state_ = State{&interpolator_};
    interpolator_.setOrder(kDegree + 1);
  }

  /// Sets a random state.
  auto setRandomState() -> void {
    const auto min_num_variables = state_.interpolator()->layout().outer_input_size;
    for (auto i = Index{0}; i < min_num_variables + Eigen::internal::random<Index>(10, 20); ++i) {
      StampedVariable stamped_variable;
      stamped_variable.stamp() = 0.25 * i;
      stamped_variable.variable() = Variable::Random();
      state_.elements().insert(stamped_variable);
    }
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(const Index degree) -> bool {
    const auto time = state_.range().sample();
    const auto result = state_.evaluate(time, degree, false);
    const auto d_result = state_.evaluate(time + kInc, degree, false);

    Tangent dx;
    for (Index i = 0; i < degree; ++i) {
      if (i == 0) {
        dx = (d_result.value - result.value) / kInc;
      } else {
        dx = (d_result.derivative(i - 1) - result.derivative(i - 1)) / kInc;
      }
      if (!dx.isApprox(result.derivative(i), kTol))
        return false;
    }

    return true;
  }

  /// Checks the Jacobians.
  /// \param degree Maximum derivative degree.
  /// \return True if numeric and analytic Jacobians are close.
  auto checkJacobians(const Index degree) -> bool {
    for (Index i = 0; i <= degree; ++i) {
      // Evaluate analytic Jacobian.
      const auto time = state_.range().sample();
      const auto result = state_.evaluate(time, i, true);

      // Retrieve inputs.
      const auto inputs = state_.variables(time);

      // Allocate Jacobian.
      Jacobian Jn_i;
      const auto num_inputs = static_cast<Index>(inputs.size());
      Jn_i.setZero(Variable::kNumParameters, num_inputs * StampedVariable::kNumParameters);

      // Evaluate Jacobian.
      for (Index j = 0; j < num_inputs; ++j) {
        auto input_j = Eigen::Map<StampedVariable>{inputs[j]->data()};

        for (Index k = 0; k < input_j.size() - 1; ++k) {
          const StampedVariable tmp = input_j;
          const Tangent tau = kInc * Tangent::Unit(k);
          input_j.variable() += tau;

          const auto d_result = state_.evaluate(time, i, false);

          if (i == 0) {
            Jn_i.col(j * StampedVariable::kNumParameters + k) = (d_result.value - result.value).transpose() / kInc;
          } else {
            Jn_i.col(j * StampedVariable::kNumParameters + k) = (d_result.derivative(i - 1) - result.derivative(i - 1)).transpose() / kInc;
          }

          input_j = tmp;
        }
      }

      // Compare Jacobians.
      if (!Jn_i.isApprox(result.jacobian(i), kTol))
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
  for (auto i = 0; i < TestFixture::kItr; ++i) {
    EXPECT_TRUE(this->checkDerivatives(TestFixture::kDegree));
  }
}

TYPED_TEST_P(CartesianStateTests, Jacobians) {
  this->setRandomState();
  for (auto i = 0; i < TestFixture::kItr; ++i) {
    EXPECT_TRUE(this->checkJacobians(TestFixture::kDegree));
  }
}

using SE3StateTestTypes = ::testing::Types<std::tuple<ContinuousState<variables::SE3<double>>, BasisInterpolator<double, Eigen::Dynamic>>>;

template <typename TArgs>
class SE3StateTests : public testing::Test {
 public:
  // Constants.
  static constexpr auto kDegree = 3;
  static constexpr auto kItr = 20;
  static constexpr auto kInc = 1e-8;
  static constexpr auto kTol = 1e-6;

  // Definitions.
  using State = typename std::tuple_element<0, TArgs>::type;
  using Interpolator = typename std::tuple_element<1, TArgs>::type;

  using Index = typename State::Index;
  using Scalar = typename State::Scalar;
  using Variable = typename State::Variable;
  using Output = typename State::Output;
  using StampedVariable = typename State::StampedVariable;

  using Tangent = variables::Tangent<Output>;
  using Jacobian = variables::JacobianX<Scalar>;

  /// Sets a random state.
  auto setRandomState() -> void {
    const auto min_num_variables = state_.interpolator()->layout().outer_input_size;
    for (auto i = Index{0}; i < min_num_variables + Eigen::internal::random<Index>(10, 20); ++i) {
      StampedVariable stamped_variable;
      stamped_variable.stamp() = 0.25 * i;
      stamped_variable.variable() = Variable::Random();
      state_.elements().insert(stamped_variable);
    }
  }

  /// Set up.
  auto SetUp() -> void final {
    state_ = State{&interpolator_};
    interpolator_.setOrder(kDegree + 1);
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(const Index degree) -> bool {
    const auto time = state_.range().sample();
    const auto result = state_.evaluate(time, degree, false);
    const auto d_result = state_.evaluate(time + kInc, degree, false);

    Tangent dx;
    for (Index i = 0; i < degree; ++i) {
      if (i == 0) {
        dx.angular() = result.value.rotation().gInv().gPlus(d_result.value.rotation()).gLog() / kInc;
        dx.linear() = (d_result.value.translation() - result.value.translation()) / kInc;
      } else {
        dx = (d_result.derivative(i - 1) - result.derivative(i - 1)) / kInc;
      }

      if (!dx.isApprox(result.derivative(i), kTol))
        return false;
    }

    return true;
  }

  /// Checks the Jacobians.
  /// \param degree Maximum derivative degree.
  /// \return True if numeric and analytic Jacobians are close.
  auto checkJacobians(const Index degree) -> bool {
    for (Index i = 0; i <= degree; ++i) {
      const auto time = state_.range().sample();
      const auto result = state_.evaluate(time, degree, true);

      // Retrieve inputs.
      const auto inputs = state_.variables(time);

      // Allocate Jacobian.
      Jacobian Jn_i;
      const auto num_inputs = static_cast<Index>(inputs.size());
      Jn_i.setZero(Tangent::kNumParameters, num_inputs * StampedVariable::kNumParameters);

      // Evaluate Jacobian.
      for (Index j = 0; j < num_inputs; ++j) {
        auto input_j = Eigen::Map<StampedVariable>{inputs[j]->data()};

        for (Index k = 0; k < Tangent::kNumParameters; ++k) {
          const StampedVariable tmp = input_j;
          const Tangent tau = kInc * Tangent::Unit(k);
          input_j.variable().rotation() *= tau.angular().gExp();
          input_j.variable().translation() += tau.linear();

          const auto d_result = state_.evaluate(time, degree, false);

          if (i == 0) {
            const auto& value = result.value;
            const auto& d_value = d_result.value;
            Jn_i.col(j * StampedVariable::kNumParameters + k).template head<3>() = (value.rotation().gInv().gPlus(d_value.rotation())).gLog() / kInc;
            Jn_i.col(j * StampedVariable::kNumParameters + k).template tail<3>() = (d_value.translation() - value.translation()) / kInc;
          } else {
            Jn_i.col(j * StampedVariable::kNumParameters + k) = (d_result.derivative(i - 1) - result.derivative(i - 1)) / kInc;
          }

          input_j = tmp;
        }

        Jn_i.template middleCols<StampedVariable::kNumParameters - 1>(j * StampedVariable::kNumParameters) =
            Jn_i.template middleCols<Tangent::kNumParameters>(j * StampedVariable::kNumParameters) * variables::JacobianAdapter<Variable>(inputs[j]->data());
      }

      // Compare Jacobians.
      if (!Jn_i.isApprox(result.jacobian(i), kTol))
        return false;
    }

    return true;
  }

 private:
  State state_;
  Interpolator interpolator_;
};

TYPED_TEST_SUITE_P(SE3StateTests);

TYPED_TEST_P(SE3StateTests, Derivatives) {
  this->setRandomState();
  for (auto i = 0; i < TestFixture::kItr; ++i) {
    EXPECT_TRUE(this->checkDerivatives(2));
  }
}

TYPED_TEST_P(SE3StateTests, Jacobians) {
  this->setRandomState();
  for (auto i = 0; i < TestFixture::kItr; ++i) {
    EXPECT_TRUE(this->checkJacobians(2));
  }
}

REGISTER_TYPED_TEST_SUITE_P(CartesianStateTests, Derivatives, Jacobians);
INSTANTIATE_TYPED_TEST_SUITE_P(HyperTests, CartesianStateTests, CartesianStateTestTypes);

REGISTER_TYPED_TEST_SUITE_P(SE3StateTests, Derivatives, Jacobians);
INSTANTIATE_TYPED_TEST_SUITE_P(HyperTests, SE3StateTests, SE3StateTestTypes);

}  // namespace hyper::state::tests
