/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/policies/forward.hpp"

namespace hyper {

class AbstractPolicy {
 public:
  /// Default destructor.
  virtual ~AbstractPolicy() = default;

  /// Collects the stamps.
  /// \param pointers Input pointers.
  /// \return Stamps.
  [[nodiscard]] virtual auto stamps(const Pointers<const Scalar>& pointers) const -> Stamps = 0;

  /// Evaluates a query.
  /// \param state_query State query.
  /// \param policy_query Policy query.
  /// \return Interpolation result.
  [[nodiscard]] virtual auto evaluate(const StateQuery& state_query, const PolicyQuery& policy_query) const -> StateResult = 0;
};

} // namespace hyper
