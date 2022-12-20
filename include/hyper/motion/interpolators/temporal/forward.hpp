/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/motion/forward.hpp"

namespace hyper {

template <typename TScalar>
class TemporalInterpolator;

template <typename TScalar, int TOrder>
class PolynomialInterpolator;

template <typename TScalar, int TOrder>
class BasisInterpolator;

template <typename TIndex>
struct TemporalInterpolatorLayout {
  const TIndex outer_input_size;
  const TIndex inner_input_size;
  const TIndex left_input_margin;
  const TIndex right_input_margin;
  const TIndex left_input_padding;
  const TIndex right_input_padding;
  const TIndex output_size;
};

template <typename TScalar>
struct TemporalInterpolatorQuery {
  TScalar time;
  MotionDerivative derivative;
  const std::vector<TScalar>& times;
  TScalar* weights;
};

} // namespace hyper
