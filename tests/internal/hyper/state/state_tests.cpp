/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <fstream>

#include <gtest/gtest.h>

#include "hyper/variables/forward.hpp"

#include "hyper/state/continuous.hpp"
#include "hyper/state/interpolators/interpolators.hpp"
#include "hyper/variables/groups/se3.hpp"

namespace hyper::state::tests {

using namespace variables;

using Interpolator = BasisInterpolator<double, 4>;
using StateTestTypes = ::testing::Types<std::tuple<ContinuousState<Position<double>>, Interpolator>, std::tuple<ContinuousState<SE3<double>>, Interpolator>,
                                        std::tuple<ContinuousState<SU2<double>>, Interpolator>>;

template <typename TArgs>
class StateTests : public testing::Test {
 public:
  // Constants.
  static constexpr auto kItr = 20;
  static constexpr auto kInc = 1e-8;
  static constexpr auto kTol = 1e-6;

  // Definitions.
  using State = typename std::tuple_element<0, TArgs>::type;
  using Interpolator = typename std::tuple_element<1, TArgs>::type;

  using Scalar = typename State::Scalar;

  using Variable = typename State::Variable;
  using VariableTangent = typename State::VariableTangent;
  using StampedVariable = typename State::StampedVariable;
  using StampedVariableTangent = typename State::StampedVariableTangent;

  using Output = typename State::Output;
  using OutputTangent = typename State::OutputTangent;

  /// Set up.
  auto SetUp() -> void final {
    auto interpolator = std::make_unique<Interpolator>();
    state_ = std::make_unique<State>(std::move(interpolator), true, JacobianType::TANGENT_TO_MANIFOLD);
    setRandomState();
  }

  /// Sets a random state.
  auto setRandomState() -> void {
    const auto num_inputs = state_->layout().outer_size;
    for (auto i = 0; i < num_inputs + Eigen::internal::random<int>(10, 20); ++i) {
      StampedVariable stamped_variable;
      stamped_variable.time() = 0.25 * i;
      stamped_variable.variable() = Variable::Random();
      state_->elements().insert(stamped_variable);
    }
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(int degree) -> void {
    const auto time = state_->range().sample();
    const auto result = state_->evaluate(time, degree);
    const auto d_result = state_->evaluate(time + kInc, degree);

    for (auto i = 0; i < degree; ++i) {
      OutputTangent tau;
      if (i == 0) {
        tau = d_result.value().tMinus(result.value()) / kInc;
      } else {
        tau = d_result.tangent(i - 1).tMinus(result.tangent(i - 1)) / kInc;
      }
      EXPECT_TRUE(tau.isApprox(result.tangent(i), kTol));
    }
  }

  /// Checks the Jacobians.
  /// \param degree Maximum derivative degree.
  /// \return True if numeric and analytic Jacobians are close.
  auto checkJacobians(int degree) -> void {
    for (auto i = 0; i <= degree; ++i) {
      // Evaluate analytic Jacobian.
      const auto time = state_->range().sample();
      auto stamped_variables = state_->parameterBlocks(time);
      const auto result = state_->evaluate(time, i, true);

      JacobianX<Scalar> Jn_i;
      const auto local_input_size = state_->localInputSize();
      Jn_i.setZero(OutputTangent::kNumParameters, stamped_variables.size() * local_input_size);

      for (auto j = std::size_t{0}; j < stamped_variables.size(); ++j) {
        for (auto k = 0; k < VariableTangent::kNumParameters; ++k) {
          auto d_input_j = Eigen::Map<StampedVariable>{stamped_variables[j]}.tPlus(kInc * StampedVariableTangent::Unit(k));

          auto tmp = stamped_variables[j];
          stamped_variables[j] = d_input_j.data();
          const auto d_result = state_->evaluate(time, i, false, stamped_variables.data());
          stamped_variables[j] = tmp;

          if (i == 0) {
            Jn_i.col(j * local_input_size + k) = d_result.value().tMinus(result.value()) / kInc;
          } else {
            Jn_i.col(j * local_input_size + k) = d_result.tangent(i - 1).tMinus(result.tangent(i - 1)) / kInc;
          }
        }

        // Convert Jacobians.
        if (state_->jacobianType() == JacobianType::TANGENT_TO_MANIFOLD || state_->jacobianType() == JacobianType::TANGENT_TO_STAMPED_MANIFOLD) {
          const auto J_a = Eigen::Map<Variable>{stamped_variables[j] + StampedVariable::kVariableOffset}.tMinusJacobian();
          Jn_i.template block<OutputTangent::kNumParameters, Variable::kNumParameters>(0, j * local_input_size + StampedVariable::kVariableOffset) =
              Jn_i.template block<OutputTangent::kNumParameters, VariableTangent::kNumParameters>(0, j * local_input_size + StampedVariable::kVariableOffset) * J_a;
        }
      }

      EXPECT_TRUE(Jn_i.isApprox(result.jacobian(i), kTol));
    }
  }

 private:
  std::unique_ptr<State> state_;
};

TYPED_TEST_SUITE_P(StateTests);

TYPED_TEST_P(StateTests, Derivatives) {
  for (auto i = 0; i < TestFixture::kItr; ++i) {
    this->checkDerivatives(2);
  }
}

TYPED_TEST_P(StateTests, Jacobians) {
  for (auto i = 0; i < TestFixture::kItr; ++i) {
    this->checkJacobians(2);
  }
}

REGISTER_TYPED_TEST_SUITE_P(StateTests, Derivatives, Jacobians);
INSTANTIATE_TYPED_TEST_SUITE_P(HyperTests, StateTests, StateTestTypes);

}  // namespace hyper::state::tests
