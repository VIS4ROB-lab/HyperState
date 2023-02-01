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
                                        std::tuple<ContinuousState<SE3<double>, Tangent<SE3<double>>>, Interpolator>, std::tuple<ContinuousState<SU2<double>>, Interpolator>>;

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

  using Input = typename State::Input;
  using InputTangent = typename State::InputTangent;
  using StampedInput = typename State::StampedInput;

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
    const auto num_inputs = state_.interpolator()->layout().outer_input_size;
    for (auto i = Index{0}; i < num_inputs + Eigen::internal::random<Index>(10, 20); ++i) {
      StampedInput stamped_input;
      stamped_input.stamp() = 0.25 * i;
      stamped_input.variable() = Input::Random();
      state_.elements().insert(stamped_input);
    }
  }

  /// Checks the derivatives.
  /// \param degree Maximum derivative degree.
  /// \return True if derivatives are correct.
  auto checkDerivatives(const Index degree) -> void {
    const auto time = state_.range().sample();
    const auto result = state_.evaluate(time, degree, false);
    const auto d_result = state_.evaluate(time + kInc, degree, false);

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
      auto inputs = state_.parameterBlocks(time);
      const auto result = state_.evaluate(time, i, true);

      JacobianX<Scalar> Jn_i;
      Jn_i.setZero(OutputTangent::kNumParameters, inputs.size() * StampedOutputTangent::kNumParameters);

      for (auto j = std::size_t{0}; j < inputs.size(); ++j) {
        for (Index k = 0; k < OutputTangent::kNumParameters; ++k) {
          const InputTangent tau = kInc * InputTangent::Unit(k);
          StampedInput stamped_input = Eigen::Map<StampedInput>{inputs[j]};
          stamped_input.variable() = stamped_input.variable().tPlus(tau);

          auto tmp = inputs[j];
          inputs[j] = stamped_input.data();
          const auto d_result = state_.evaluate(time, i, false, inputs.data());
          inputs[j] = tmp;

          if (i == 0) {
            Jn_i.col(j * StampedOutputTangent::kNumParameters + k) = d_result.value().tMinus(result.value()) / kInc;
          } else {
            Jn_i.col(j * StampedOutputTangent::kNumParameters + k) = (d_result.tangent(i - 1) - result.tangent(i - 1)).transpose() / kInc;
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
