/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/motion/interpolators/spatial/forward.hpp"
#include "hyper/motion/interpolators/temporal/forward.hpp"

#include "hyper/variables/groups/se3.hpp"

namespace hyper {

template <>
class SpatialInterpolator<Stamped<SE3<Scalar>>> final {
 public:
  // Definitions.
  using Input = Stamped<SE3<Scalar>>;

  // Definitions.
  using Index = Eigen::Index;
  // using Scalar = typename SE3<Scalar>::Scalar;

  using Manifold = SE3<Scalar>;
  using Tangent = hyper::Tangent<SE3<Scalar>>;

  // Constants.
  static constexpr auto kDimManifold = Manifold::kNumParameters;
  static constexpr auto kDimTangent = Tangent::kNumParameters;

  /// Evaluates this.
  /// \param query Temporal motion query.
  /// \param layout Temporal interpolator layout.
  /// \param weights Weights.
  /// \param inputs Inputs.
  /// \return Temporal motion results.
  [[nodiscard]] static auto evaluate(const TemporalMotionQuery<Scalar>& query,
      const TemporalInterpolatorLayout<Index>& layout,
      const Eigen::Ref<const MatrixX<Scalar>>& weights,
      const Scalar* const* inputs) -> TemporalMotionResult<Scalar>;

 private:
  template <MotionDerivative TMotionDerivative>
  [[nodiscard]] static auto evaluate(const TemporalMotionQuery<Scalar>& query,
      const TemporalInterpolatorLayout<Index>& layout,
      const Eigen::Ref<const MatrixX<Scalar>>& weights,
      const Scalar* const* inputs) -> TemporalMotionResult<Scalar>;
};

} // namespace hyper
