/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/forward.hpp"

#include "hyper/variables/groups/su2.hpp"

namespace hyper::state {

template <typename TScalar>
class SpatialInterpolator<variables::SU2<TScalar>> final {
 public:
  // Definitions.
  using Index = Eigen::Index;
  using Input = variables::SU2<TScalar>;
  using Output = variables::SU2<TScalar>;

  /// Evaluates this.
  static auto evaluate(Result<Output>& result, const Eigen::Ref<const MatrixX<TScalar>>& weights, const TScalar* const* inputs, const Index& s_idx, const Index& e_idx,
                       const Index& offs) -> void;
};

}  // namespace hyper::state
