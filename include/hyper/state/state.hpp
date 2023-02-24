/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/forward.hpp"

#include "hyper/state/definitions.hpp"

namespace hyper::state {

template <typename TScalar>
class State {
 public:
  // Definitions.
  using Scalar = TScalar;

  /// Virtual default destructor.
  virtual ~State() = default;
};

}  // namespace hyper::state
