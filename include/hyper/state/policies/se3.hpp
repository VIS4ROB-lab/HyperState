/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/policies/abstract.hpp"
#include "hyper/variables/groups/forward.hpp"

namespace hyper {

template <>
class ManifoldPolicy<Stamped<SE3<Scalar>>> final : public AbstractPolicy {
 public:
  // Definitions.
  using Value = SE3<Scalar>;
  using Input = Stamped<Value>;
  using Derivative = Tangent<Value>;

  /// Collects the stamps.
  /// \return Stamps.
  [[nodiscard]] auto stamps(const Pointers<const Scalar>& pointers) const -> Stamps final;

  /// Evaluates a query.
  /// \param state_query State query.
  /// \param policy_query Policy query.
  /// \return Interpolation result.
  [[nodiscard]] auto evaluate(const StateQuery& state_query, const PolicyQuery& policy_query) const -> StateResult final;
};

} // namespace hyper
