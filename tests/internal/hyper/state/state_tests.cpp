/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <fstream>

#include <gtest/gtest.h>

#include "hyper/variables/forward.hpp"

#include "hyper/state/continuous.hpp"
#include "hyper/state/interpolators/interpolators.hpp"

namespace hyper::state::tests {

using namespace variables;

using Interpolator = BasisInterpolator<double, 4>;
using StateTestTypes = ::testing::Types<std::tuple<ContinuousState<Position<double>>, Interpolator>, std::tuple<ContinuousState<SE3<double>>, Interpolator>,
                                        std::tuple<ContinuousState<SU2<double>>, Interpolator>>;

template <typename TArgs>
class StateTests : public testing::Test {
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
  using VariableTangent = typename State::VariableTangent;
  using StampedVariable = typename State::StampedVariable;
  using StampedVariableTangent = typename State::StampedVariableTangent;

  using OutputTangent = typename State::OutputTangent;
  using StampedOutputTangent = typename State::StampedOutputTangent;

  /// Set up.
  auto SetUp() -> void final {
    state_ = State{&interpolator_};
    interpolator_.setOrder(kDegree + 1);
    setRandomState();
  }

  /// Sets a random state.
  auto setRandomState() -> void {
    const auto num_inputs = state_.interpolator()->layout(state_.isUniform()).outer_input_size;
    for (auto i = Index{0}; i < num_inputs + Eigen::internal::random<Index>(10, 20); ++i) {
      StampedVariable stamped_variable;
      stamped_variable.time() = 0.25 * i;
      stamped_variable.variable() = Variable::Random();
      state_.elements().insert(stamped_variable);
    }
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(const Index degree) -> void {
    const auto time = state_.range().sample();
    const auto result = state_.evaluate(time, degree);
    const auto d_result = state_.evaluate(time + kInc, degree);

    for (Index i = 0; i < degree; ++i) {
      OutputTangent tau;
      if (i == 0) {
        tau = d_result.value().tMinus(result.value()) / kInc;
      } else {
        tau = (d_result.tangent(i - 1) - result.tangent(i - 1)) / kInc;
      }
      EXPECT_TRUE(tau.isApprox(result.tangent(i), kTol));
    }
  }

  /// Checks the Jacobians.
  /// \param degree Maximum derivative degree.
  /// \return True if numeric and analytic Jacobians are close.
  auto checkJacobians(const Index degree) -> void {
    for (Index i = 0; i <= degree; ++i) {
      // Evaluate analytic Jacobian.
      const auto time = state_.range().sample();
      auto stamped_variables = state_.parameterBlocks(time);
      const auto result = state_.evaluate(time, i, JacobianType::TANGENT_TO_TANGENT);

      JacobianX<Scalar> Jn_i;
      if (state_.isUniform()) {
        Jn_i.setZero(OutputTangent::kNumParameters, stamped_variables.size() * OutputTangent::kNumParameters);
      } else {
        Jn_i.setZero(OutputTangent::kNumParameters, stamped_variables.size() * StampedOutputTangent::kNumParameters);
      }

      for (auto j = std::size_t{0}; j < stamped_variables.size(); ++j) {
        for (Index k = 0; k < OutputTangent::kNumParameters; ++k) {
          auto d_input_j = Eigen::Map<StampedVariable>{stamped_variables[j]}.tPlus(kInc * StampedVariableTangent::Unit(k));

          auto tmp = stamped_variables[j];
          stamped_variables[j] = d_input_j.data();
          const auto d_result = state_.evaluate(time, i, JacobianType::NONE, stamped_variables.data());
          stamped_variables[j] = tmp;

          if (state_.isUniform()) {
            if (i == 0) {
              Jn_i.col(j * OutputTangent::kNumParameters + k) = d_result.value().tMinus(result.value()) / kInc;
            } else {
              Jn_i.col(j * OutputTangent::kNumParameters + k) = (d_result.tangent(i - 1) - result.tangent(i - 1)).transpose() / kInc;
            }
          } else {
            if (i == 0) {
              Jn_i.col(j * StampedOutputTangent::kNumParameters + k) = d_result.value().tMinus(result.value()) / kInc;
            } else {
              Jn_i.col(j * StampedOutputTangent::kNumParameters + k) = (d_result.tangent(i - 1) - result.tangent(i - 1)).transpose() / kInc;
            }
          }
        }
      }

      EXPECT_TRUE(Jn_i.isApprox(result.jacobian(i), kTol));
    }
  }

 private:
  State state_;
  Interpolator interpolator_;
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
