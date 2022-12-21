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
  /// \param query Temporal motion query.
  /// \param layout Temporal interpolator layout.
  /// \param weights Weights.
  /// \param inputs Inputs.
  /// \return Temporal motion results.
  [[nodiscard]] static auto evaluate(
      const TemporalMotionQuery<Scalar>& query,
      const TemporalInterpolatorLayout<Index>& layout,
      const Eigen::Ref<const MatrixX<Scalar>>& weights,
      const Scalar* const* inputs) -> TemporalMotionResult<Scalar> {
    // Unpack queries.
    const auto& [stamp, derivative, jacobian] = query;

    // Sanity checks.
    DCHECK(weights.rows() == layout.inner_input_size && weights.cols() == derivative + 1);

    // Allocate result.
    TemporalMotionResult<Scalar> result;
    auto& [derivatives, jacobians] = result;
    derivatives.reserve(derivative + 1);
    jacobians.reserve(derivative + 1);

    if (layout.outer_input_size == 1) {
      for (Index k = 0; k <= derivative; ++k) {
        if (k == 0) {
          derivatives.emplace_back(Eigen::Map<const Input>{inputs[0]}.variable());
          if (jacobian) {
            jacobians.emplace_back(JacobianX<Scalar>::Identity(kDimTangent, kNumInputParameters));
          }
        } else {
          derivatives.emplace_back(MatrixX<Scalar>::Zero(kDimManifold, 1));
          if (jacobian) {
            jacobians.emplace_back(JacobianX<Scalar>::Zero(kDimTangent, kNumInputParameters));
          }
        }
      }
    } else {
      // Compute indices.
      const auto start_idx = layout.left_input_padding;
      const auto end_idx = start_idx + layout.inner_input_size - 1;

      // Evaluate increments.
      using Increments = Eigen::Matrix<Scalar, kDimTangent, Eigen::Dynamic>;
      auto increments = Increments{kDimTangent, end_idx - start_idx + 1};
      increments.col(0).noalias() = Eigen::Map<const Input>{inputs[start_idx]}.variable();

      for (auto i = start_idx; i < end_idx; ++i) {
        const auto x = Eigen::Map<const Input>{inputs[i]};
        const auto y = Eigen::Map<const Input>{inputs[i + 1]};
        increments.col(i - start_idx + 1).noalias() = y.variable() - x.variable();
      }

      for (Index k = 0; k <= derivative; ++k) {
        derivatives.emplace_back(increments * weights.col(k));

        if (jacobian) {
          jacobians.emplace_back(JacobianX<Scalar>::Zero(kDimTangent, layout.outer_input_size * kNumInputParameters));

          if (k == 0) {
            jacobians[k].template middleCols<kNumInputParameters>(start_idx * kNumInputParameters).diagonal().setConstant(Scalar{1} - weights(1, k));
          } else {
            jacobians[k].template middleCols<kNumInputParameters>(start_idx * kNumInputParameters).diagonal().setConstant(Scalar{-1} * weights(1, k));
          }

          for (auto j = start_idx + 1; j < end_idx; ++j) {
            jacobians[k].template middleCols<kNumInputParameters>(j * kNumInputParameters).diagonal().setConstant(weights(j, k) - weights(j + 1, k));
          }

          jacobians[k].template middleCols<kNumInputParameters>(end_idx * kNumInputParameters).diagonal().setConstant(weights(end_idx, k));
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
