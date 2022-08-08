/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/forward.hpp"

namespace hyper {

struct InterpolatorLayout {
  IndexRange outer; ///< Outer index range (including padding).
  IndexRange inner; ///< Inner index range (excluding padding).
};

class AbstractInterpolator;

class PolynomialInterpolator;

class BasisInterpolator;

} // namespace hyper
