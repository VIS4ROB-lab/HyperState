/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/forward.hpp"

namespace hyper::state {

struct TemporalInterpolatorLayout {
  int outer_size;
  int inner_size;
  int left_margin;
  int right_margin;
  int left_padding;
  int right_padding;
};

class TemporalInterpolator;

template <int TOrder>
class PolynomialInterpolator;

template <int TOrder>
class BasisInterpolator;

}  // namespace hyper::state
