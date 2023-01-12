/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/motion/forward.hpp"

namespace hyper {

template <typename TScalar>
class Motion {
 public:
  // Definitions.
  using Scalar = TScalar;

  using Index = Eigen::Index;

  /// Virtual default destructor.
  virtual ~Motion() = default;
};

} // namespace hyper
