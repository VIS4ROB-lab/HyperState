/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/motion/interpolators/temporal/forward.hpp"

namespace hyper {

template <typename TScalar>
class TemporalInterpolator {
 public:
  // Definitions.
  using Scalar = TScalar;

  using Index = Eigen::Index;
  using Layout = TemporalInterpolatorLayout<Index>;
  using Query = TemporalInterpolatorQuery<Scalar>;

  /// Default destructor.
  virtual ~TemporalInterpolator() = default;

  /// Retrieves the layout.
  /// \return Layout.
  [[nodiscard]] virtual auto layout() const -> Layout = 0;

  /// Evaluates this.
  /// \param query Query.
  /// \return True on success.
  virtual auto evaluate(const Query& query) const -> bool = 0;
};

} // namespace hyper
