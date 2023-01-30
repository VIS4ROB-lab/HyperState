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

  using Scalar = TScalar;
  using Input = variables::SU2<TScalar>;
  using Output = variables::SU2<TScalar>;

  /// Evaluates this.
  static auto evaluate(const Index& derivative, const Scalar* const* inputs, const Index& num_inputs, const Index& start_index, const Index& end_index,
                       const Index& num_input_parameters, const Index& input_offset, const Eigen::Ref<const MatrixX<Scalar>>& weights, bool jacobians) -> Result<Output>;
};

template <typename TScalar>
class SpatialInterpolator<variables::SU2<TScalar>, variables::Tangent<variables::SU2<TScalar>>> final {
 public:
  // Definitions.
  using Index = Eigen::Index;

  using Scalar = TScalar;
  using Input = variables::Tangent<variables::SU2<TScalar>>;
  using Output = variables::SU2<TScalar>;

  /// Evaluates this.
  static auto evaluate(const Index& derivative, const Scalar* const* inputs, const Index& num_inputs, const Index& start_index, const Index& end_index,
                       const Index& num_input_parameters, const Index& input_offset, const Eigen::Ref<const MatrixX<Scalar>>& weights, bool jacobians) -> Result<Output>;
};

}  // namespace hyper::state
