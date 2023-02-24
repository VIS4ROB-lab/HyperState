/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/forward.hpp"

#include "hyper/state/definitions.hpp"

namespace hyper::state {

template <typename TScalar>
class TemporalInterpolator;

template <typename TScalar, int TOrder>
class PolynomialInterpolator;

template <typename TScalar, int TOrder>
class BasisInterpolator;

struct TemporalInterpolatorLayout {
  Index outer_size;
  Index inner_size;
  Index left_margin;
  Index right_margin;
  Index left_padding;
  Index right_padding;
};

}  // namespace hyper::state
