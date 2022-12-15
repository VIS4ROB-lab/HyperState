/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/forward.hpp"

namespace hyper {

class AbstractInterpolator;

class PolynomialInterpolator;

class BasisInterpolator;

template<typename TIndex>
struct TemporalInterpolatorLayout {
  const TIndex outer_input_size;
  const TIndex inner_input_size;
  const TIndex left_input_margin;
  const TIndex right_input_margin;
  const TIndex left_input_padding;
  const TIndex right_input_padding;
  const TIndex output_size;
};

} // namespace hyper
