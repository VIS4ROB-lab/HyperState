/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/forward.hpp"

namespace hyper {

template <typename TScalar>
class AbstractInterpolator {
 public:
  // Definitions.
  using Scalar = TScalar;
  using Layout = TemporalInterpolatorLayout<Eigen::Index>;
  using Query = TemporalInterpolatorQuery<TScalar, Eigen::Index>;

  /// Default destructor.
  virtual ~AbstractInterpolator() = default;

  /// Retrieves the layout.
  /// \return Layout.
  [[nodiscard]] virtual auto layout() const -> Layout = 0;

  /// Evaluates this.
  /// \param query Query.
  /// \return True on success.
  [[nodiscard]] virtual auto evaluate(const Query& query) const -> bool = 0;
};

} // namespace hyper
