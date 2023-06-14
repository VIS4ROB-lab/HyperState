/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <fstream>

#include <gtest/gtest.h>

#include "hyper/variables/forward.hpp"

#include "hyper/state/continuous.hpp"
#include "hyper/state/interpolators/interpolators.hpp"
#include "hyper/variables/se3.hpp"

namespace hyper::state::tests {

using namespace variables;

using Interpolator = BasisInterpolator<4>;
using StateTestTypes =
    ::testing::Types<std::tuple<ContinuousState<R3>, Interpolator>, std::tuple<ContinuousState<SU2>, Interpolator>, std::tuple<ContinuousState<SE3>, Interpolator>>;

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

  using Group = typename State::Group;
  using GroupTangent = variables::Tangent<Group>;
  using StampedGroup = variables::Stamped<Group>;
  using StampedGroupTangent = variables::Stamped<GroupTangent>;

  /// Set up.
  auto SetUp() -> void final {
    auto interpolator = std::make_unique<Interpolator>();
    state_ = std::make_unique<State>(std::move(interpolator), true, Jacobian::TANGENT_TO_GROUP);
    setRandomState();
  }

  /// Sets a random state.
  auto setRandomState() -> void {
    const auto num_inputs = state_->layout().outer_size;
    for (auto i = 0; i < num_inputs + Eigen::internal::random<int>(10, 20); ++i) {
      StampedGroup stamped_parameter;
      stamped_parameter.time() = 0.25 * i;
      stamped_parameter.variable() = Group::Random();
      state_->stampedParameters().insert(stamped_parameter);
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
      GroupTangent tau;
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
      const auto result = state_->evaluate(time, i, true);
      auto parameter_blocks = state_->parameterBlocks(time);

      JacobianX Jn_i;
      const auto tangent_input_size = state_->tangentInputSize();
      Jn_i.setZero(GroupTangent::kNumParameters, parameter_blocks.size() * tangent_input_size);

      for (auto j = std::size_t{0}; j < parameter_blocks.size(); ++j) {
        for (auto k = 0; k < GroupTangent::kNumParameters; ++k) {
          auto d_input_j = Eigen::Map<StampedGroup>{parameter_blocks[j]}.tPlus(kInc * StampedGroupTangent::Unit(k));

          auto tmp = parameter_blocks[j];
          parameter_blocks[j] = d_input_j.data();
          const auto d_result = state_->evaluate(time, i, false, parameter_blocks.data());
          parameter_blocks[j] = tmp;

          if (i == 0) {
            Jn_i.col(j * tangent_input_size + k) = d_result.value().tMinus(result.value()) / kInc;
          } else {
            Jn_i.col(j * tangent_input_size + k) = d_result.tangent(i - 1).tMinus(result.tangent(i - 1)) / kInc;
          }
        }

        // Convert Jacobians.
        if (state_->jacobian() == Jacobian::TANGENT_TO_GROUP || state_->jacobian() == Jacobian::TANGENT_TO_STAMPED_GROUP) {
          const auto J_a = Eigen::Map<Group>{parameter_blocks[j] + StampedGroup::kVariableOffset}.tMinusJacobian();
          Jn_i.template block<GroupTangent::kNumParameters, Group::kNumParameters>(0, j * tangent_input_size + StampedGroup::kVariableOffset) =
              Jn_i.template block<GroupTangent::kNumParameters, GroupTangent::kNumParameters>(0, j * tangent_input_size + StampedGroup::kVariableOffset) * J_a;
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
