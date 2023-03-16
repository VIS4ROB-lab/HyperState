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
  int outer_size;
  int inner_size;
  int left_margin;
  int right_margin;
  int left_padding;
  int right_padding;
};

}  // namespace hyper::state
