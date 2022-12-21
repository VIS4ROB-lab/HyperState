/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/motion/forward.hpp"
#include "hyper/motion/interpolators/spatial/forward.hpp"
#include "hyper/motion/interpolators/temporal/forward.hpp"

#include "hyper/variables/cartesian.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper {

template <typename TVariable>
class SpatialInterpolator<Stamped<TVariable>> final {
 public:
  // Definitions.
  using Input = Stamped<TVariable>;

  // Definitions.
  using Index = Eigen::Index;
  using Scalar = typename TVariable::Scalar;

  using Manifold = TVariable;
  using Tangent = hyper::Tangent<TVariable>;

  // Constants.
  static constexpr auto kDimManifold = Manifold::kNumParameters;
  static constexpr auto kDimTangent = Tangent::kNumParameters;

  /// Evaluates this.
  /// \param weights Interpolation weights.
  /// \param variables Interpolation variables.
  /// \param offset Offset into variables.
  /// \param jacobians Jacobians evaluation flag.
  /// \return Temporal motion results.
  [[nodiscard]] static auto evaluate(
      const Eigen::Ref<const MatrixX<Scalar>>& weights,
      const Pointers<const Scalar>& variables,
      const Index& offset,
      const bool jacobians) -> TemporalMotionResult<Scalar> {
    // Definitions.
    using Increments = Eigen::Matrix<Scalar, kDimTangent, Eigen::Dynamic>;

    const auto num_variables = weights.rows();
    const auto num_derivatives = weights.cols();

    // Allocate result.
    TemporalMotionResult<Scalar> result;
    auto& [xs, Js] = result;
    xs.reserve(num_derivatives);
    Js.reserve(num_derivatives);

    if (variables.size() == 1) {
      for (Index k = 0; k < num_derivatives; ++k) {
        if (k == 0) {
          xs.emplace_back(Eigen::Map<const Input>{variables[0]}.variable());
          if (jacobians) {
            Js.emplace_back(JacobianX<Scalar>::Identity(kDimTangent, kNumInputParameters));
          }
        } else {
          xs.emplace_back(MatrixX<Scalar>::Zero(kDimManifold, 1));
          if (jacobians) {
            Js.emplace_back(JacobianX<Scalar>::Zero(kDimTangent, kNumInputParameters));
          }
        }
      }
    } else {
      // Compute indices.
      const auto end_idx = offset + num_variables;
      const auto last_idx = end_idx - 1;

      // Compute increments.
      auto increments = Increments{kDimTangent, num_variables};
      increments.col(0).noalias() = Eigen::Map<const Input>{variables[offset]}.variable();

      for (auto i = offset + 1; i < end_idx; ++i) {
        const auto x = Eigen::Map<const Input>{variables[i - 1]};
        const auto y = Eigen::Map<const Input>{variables[i]};
        increments.col(i - offset).noalias() = y.variable() - x.variable();
      }

      for (Index k = 0; k < num_derivatives; ++k) {
        xs.emplace_back(increments * weights.col(k));

        if (jacobians) {
          Js.emplace_back(JacobianX<Scalar>::Zero(kDimTangent, variables.size() * kNumInputParameters));

          if (k == 0) {
            Js[k].template middleCols<kNumInputParameters>(offset * kNumInputParameters).diagonal().setConstant(Scalar{1} - weights(1, k));
          } else {
            Js[k].template middleCols<kNumInputParameters>(offset * kNumInputParameters).diagonal().setConstant(Scalar{-1} * weights(1, k));
          }

          for (auto j = offset + 1; j < last_idx; ++j) {
            Js[k].template middleCols<kNumInputParameters>(j * kNumInputParameters).diagonal().setConstant(weights(j, k) - weights(j + 1, k));
          }

          Js[k].template middleCols<kNumInputParameters>(last_idx * kNumInputParameters).diagonal().setConstant(weights(last_idx, k));
        }
      }
    }

    return result;
  }

 private:
  // Constants.
  static constexpr auto kNumInputParameters = Input::kNumParameters;
};

} // namespace hyper
