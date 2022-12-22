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
  static auto evaluate(const Weights& weights, const Variables& variables, const Outputs& outputs, const Jacobians& jacobians, const Index& offset, const bool old_jacobians) -> bool {
    // Definitions.
    using Increments = Eigen::Matrix<Scalar, kDimTangent, Eigen::Dynamic>;

    const auto num_variables = weights.rows();
    const auto num_derivatives = weights.cols();

    if (variables.size() == 1) {
      for (Index k = 0; k < num_derivatives; ++k) {
        if (k == 0) {
          Eigen::Map<TVariable>{outputs[0]} = Eigen::Map<const TVariable>{variables[0]};
          if (old_jacobians) {
            Eigen::Map<JacobianNM<Tangent, TVariable>>{jacobians[0][0]}.setIdentity();
          }
        } else {
          Eigen::Map<Tangent>{outputs[k]}.setZero();
          if (old_jacobians) {
            Eigen::Map<JacobianNM<Tangent, TVariable>>{jacobians[k][0]}.setZero();
          }
        }
      }
    } else {
      // Compute indices.
      const auto end_idx = offset + num_variables;
      const auto last_idx = end_idx - 1;

      // Compute increments.
      auto increments = Increments{kDimTangent, num_variables};
      increments.col(0).noalias() = Eigen::Map<const Manifold>{variables[offset]};

      for (auto i = offset + 1; i < end_idx; ++i) {
        increments.col(i - offset).noalias() = Eigen::Map<const Manifold>{variables[i]} - Eigen::Map<const Manifold>{variables[i - 1]};
      }

      for (Index k = 0; k < num_derivatives; ++k) {
        if (k == 0) {
          Eigen::Map<Manifold>{outputs[0]} = increments * weights.col(0);
        } else {
          Eigen::Map<Tangent>{outputs[k]} = increments * weights.col(k);
        }

        if (old_jacobians) {
          if (k == 0) {
            Eigen::Map<JacobianNM<Tangent, Manifold>>{jacobians[0][offset]}.diagonal().setConstant(Scalar{1} - weights(1, k));
          } else {
            Eigen::Map<JacobianNM<Tangent, Manifold>>{jacobians[k][offset]}.diagonal().setConstant(Scalar{-1} * weights(1, k));
          }

          for (auto j = offset + 1; j < last_idx; ++j) {
            Eigen::Map<JacobianNM<Tangent, Manifold>>{jacobians[k][j]}.diagonal().setConstant(weights(j, k) - weights(j + 1, k));
          }

          Eigen::Map<JacobianNM<Tangent, Manifold>>{jacobians[k][last_idx]}.diagonal().setConstant(weights(last_idx, k));
        }
      }
    }

    return true;
  }
};

} // namespace hyper
