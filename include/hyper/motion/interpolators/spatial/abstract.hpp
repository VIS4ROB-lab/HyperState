/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "forward.hpp"
#include "hyper/motion/forward.hpp"

namespace hyper {

class AbstractPolicy {
 public:
  /// Default destructor.
  virtual ~AbstractPolicy() = default;

  /// Collects the times.
  /// \param pointers Input pointers.
  /// \return Times.
  [[nodiscard]] virtual auto times(const Pointers<const Scalar>& pointers) const -> Times = 0;

  /// Evaluates a query.
  /// \param state_query State query.
  /// \param policy_query Policy query.
  /// \return Interpolation result.
  [[nodiscard]] virtual auto evaluate(const StateQuery& state_query, const PolicyQuery& policy_query) const -> StateResult = 0;
};

} // namespace hyper
