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
  /// \param layout Temporal interpolator layout.
  /// \param weights Interpolation weights.
  /// \param variables Interpolation variables.
  /// \param jacobians Jacobians evaluation flag.
  /// \return Temporal motion results.
  [[nodiscard]] static auto evaluate(
      const TemporalInterpolatorLayout<Index>& layout,
      const Eigen::Ref<const MatrixX<Scalar>>& weights,
      const Pointers<const Scalar>& variables,
      const bool jacobians) -> TemporalMotionResult<Scalar> {
    // Sanity checks.
    DCHECK_EQ(weights.rows(), layout.inner_input_size);
    DCHECK_EQ(variables.size(), layout.outer_input_size);
    const auto num_derivatives = weights.cols();

    // Allocate result.
    TemporalMotionResult<Scalar> result;
    auto& [xs, Js] = result;
    xs.reserve(num_derivatives);
    Js.reserve(num_derivatives);

    if (layout.outer_input_size == 1) {
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
      const auto start_idx = layout.left_input_padding;
      const auto end_idx = start_idx + layout.inner_input_size - 1;

      // Evaluate increments.
      using Increments = Eigen::Matrix<Scalar, kDimTangent, Eigen::Dynamic>;
      auto increments = Increments{kDimTangent, end_idx - start_idx + 1};
      increments.col(0).noalias() = Eigen::Map<const Input>{variables[start_idx]}.variable();

      for (auto i = start_idx; i < end_idx; ++i) {
        const auto x = Eigen::Map<const Input>{variables[i]};
        const auto y = Eigen::Map<const Input>{variables[i + 1]};
        increments.col(i - start_idx + 1).noalias() = y.variable() - x.variable();
      }

      for (Index k = 0; k < num_derivatives; ++k) {
        xs.emplace_back(increments * weights.col(k));

        if (jacobians) {
          Js.emplace_back(JacobianX<Scalar>::Zero(kDimTangent, layout.outer_input_size * kNumInputParameters));

          if (k == 0) {
            Js[k].template middleCols<kNumInputParameters>(start_idx * kNumInputParameters).diagonal().setConstant(Scalar{1} - weights(1, k));
          } else {
            Js[k].template middleCols<kNumInputParameters>(start_idx * kNumInputParameters).diagonal().setConstant(Scalar{-1} * weights(1, k));
          }

          for (auto j = start_idx + 1; j < end_idx; ++j) {
            Js[k].template middleCols<kNumInputParameters>(j * kNumInputParameters).diagonal().setConstant(weights(j, k) - weights(j + 1, k));
          }

          Js[k].template middleCols<kNumInputParameters>(end_idx * kNumInputParameters).diagonal().setConstant(weights(end_idx, k));
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
