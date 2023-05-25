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

using Interpolator = BasisInterpolator<double, 4>;
using StateTestTypes = ::testing::Types<std::tuple<ContinuousState<R3<double>>, Interpolator>, std::tuple<ContinuousState<SU2<double>>, Interpolator>,
                                        std::tuple<ContinuousState<SE3<double>>, Interpolator>>;

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

  using Element = typename State::Element;
  using ElementTangent = typename State::ElementTangent;
  using StampedElement = typename State::StampedElement;
  using StampedElementTangent = typename State::StampedElementTangent;

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
      StampedElement stamped_element;
      stamped_element.time() = 0.25 * i;
      stamped_element.variable() = Element::Random();
      state_->stampedElements().insert(stamped_element);
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
      ElementTangent tau;
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

      JacobianX<Scalar> Jn_i;
      const auto tangent_input_size = state_->tangentInputSize();
      Jn_i.setZero(ElementTangent::kNumParameters, parameter_blocks.size() * tangent_input_size);

      for (auto j = std::size_t{0}; j < parameter_blocks.size(); ++j) {
        for (auto k = 0; k < ElementTangent::kNumParameters; ++k) {
          auto d_input_j = Eigen::Map<StampedElement>{parameter_blocks[j]}.tPlus(kInc * StampedElementTangent::Unit(k));

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
        if (state_->jacobianType() == JacobianType::TANGENT_TO_MANIFOLD || state_->jacobianType() == JacobianType::TANGENT_TO_STAMPED_MANIFOLD) {
          const auto J_a = Eigen::Map<Element>{parameter_blocks[j] + StampedElement::kVariableOffset}.tMinusJacobian();
          Jn_i.template block<ElementTangent::kNumParameters, Element::kNumParameters>(0, j * tangent_input_size + StampedElement::kVariableOffset) =
              Jn_i.template block<ElementTangent::kNumParameters, ElementTangent::kNumParameters>(0, j * tangent_input_size + StampedElement::kVariableOffset) * J_a;
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
