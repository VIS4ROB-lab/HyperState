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
class SpatialInterpolator final {
 public:
  // Definitions.
  using Index = Eigen::Index;
  using Scalar = typename TVariable::Scalar;

  using Manifold = TVariable;
  using Tangent = hyper::Tangent<TVariable>;

  using Variables = Pointers<const Scalar>;
  using Weights = Eigen::Ref<const MatrixX<Scalar>>;
  using Outputs = Pointers<Scalar>;
  using Jacobians = std::vector<Pointers<Scalar>>;

  // Constants.
  static constexpr auto kDimManifold = Manifold::kNumParameters;
  static constexpr auto kDimTangent = Tangent::kNumParameters;

  /// Evaluates this.
  /// \param weights Interpolation weights.
  /// \param variables Interpolation variables.
  /// \param offset Offset into variables.
  /// \param jacobians Jacobians evaluation flag.
  /// \return Temporal motion results.
  static auto evaluate(const Weights& weights, const Variables& variables, const Outputs& outputs, const Jacobians* jacobians, const Index& offset) -> bool {
    // Definitions.
    using Increments = Eigen::Matrix<Scalar, kDimTangent, Eigen::Dynamic>;
    using Jacobian = JacobianNM<Tangent, Manifold>;

    // Constants.
    const auto num_variables = weights.rows();
    const auto num_derivatives = weights.cols();
    const auto last_idx = offset + num_variables - 1;

    // Compute increments.
    auto increments = Increments{kDimTangent, num_variables};
    increments.col(0).noalias() = Eigen::Map<const Manifold>{variables[offset]};

    for (auto i = offset; i < last_idx; ++i) {
      increments.col(i - offset + 1).noalias() = Eigen::Map<const Manifold>{variables[i + 1]} - Eigen::Map<const Manifold>{variables[i]};
    }

    for (Index k = 0; k < num_derivatives; ++k) {
      // Evaluate values.
      if (k == 0) {
        Eigen::Map<Manifold>{outputs[0]} = increments * weights.col(0);
      } else {
        Eigen::Map<Tangent>{outputs[k]} = increments * weights.col(k);
      }

      // Evaluate Jacobians.
      if (jacobians) {
        const auto& Js = *jacobians;
        if (variables.size() > 1) {
          if (k == 0) {
            Eigen::Map<Jacobian>{Js[0][offset]}.diagonal().setConstant(Scalar{1} - weights(1, k));
          } else {
            Eigen::Map<Jacobian>{Js[k][offset]}.diagonal().setConstant(Scalar{-1} * weights(1, k));
          }
          for (auto i = offset + 1; i < last_idx; ++i) {
            Eigen::Map<Jacobian>{Js[k][i]}.diagonal().setConstant(weights(i, k) - weights(i + 1, k));
          }
          Eigen::Map<Jacobian>{Js[k][last_idx]}.diagonal().setConstant(weights(last_idx, k));
        } else if (k == 0) {
          Eigen::Map<Jacobian>{Js[0][offset]}.diagonal().setConstant(Scalar{1});
        }
      }
    }

    return true;
  }
};

} // namespace hyper
