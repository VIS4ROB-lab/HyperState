/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/forward.hpp"

#include "hyper/variables/groups/se3.hpp"

namespace hyper::state {

template <typename TScalar>
class SpatialInterpolator<variables::SE3<TScalar>> final {
 public:
  // Definitions.
  using Index = Eigen::Index;
  using Input = variables::SE3<TScalar>;
  using Output = variables::SE3<TScalar>;

  /// Evaluates this.
  static auto evaluate(Result<Output>& result, const Eigen::Ref<const MatrixX<TScalar>>& weights, const TScalar* const* inputs, const Index& s_idx, const Index& e_idx,
                       const Index& offs) -> void;
};

template <typename TScalar>
class SpatialInterpolator<variables::SE3<TScalar>, variables::Tangent<variables::SE3<TScalar>>> final {
 public:
  // Definitions.
  using Index = Eigen::Index;
  using Input = variables::Tangent<variables::SE3<TScalar>>;
  using Output = variables::SE3<TScalar>;

  /// Evaluates this.
  static auto evaluate(Result<Output>& result, const Eigen::Ref<const MatrixX<TScalar>>& weights, const TScalar* const* inputs, const Index& s_idx, const Index& e_idx,
                       const Index& offs) -> void;
};

}  // namespace hyper::state
