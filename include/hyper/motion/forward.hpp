/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <utility>
#include <vector>

#include <glog/logging.h>

#include "hyper/definitions.hpp"

namespace hyper {

enum MotionDerivative {
  VALUE = 0,
  VELOCITY = 1,
  ACCELERATION = 2,
  JERK = 3,
};

template <typename TScalar>
class Motion;

template <typename TLabel, typename TVariable>
class LabeledMotion;

template <typename TVariable>
class TemporalMotion;

template <typename TVariable>
class DiscreteMotion;

template <typename TVariable>
class ContinuousMotion;

template <typename TScalar>
struct TemporalMotionResult {
  // Definitions.
  using Derivative = Eigen::Matrix<TScalar, Eigen::Dynamic, 1>;
  using Jacobian = Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Derivatives = std::vector<Derivative>;
  using Jacobians = std::vector<Jacobian>;

  /// Value accessor.
  /// \tparam TDerived Derived type.
  /// \param index Input index.
  /// \return Value.
  template <typename TDerived>
  inline auto derivativeAs(const Index index) const {
    DCHECK_LT(index, derivatives.size());
    return Eigen::Map<const TDerived>{derivatives[index].data()};
  }

  /// Value accessor.
  /// \tparam TDerived Derived type.
  /// \param index Input index.
  /// \return Value.
  template <typename TDerived>
  inline auto derivativeAs(const Index index) {
    DCHECK_LT(index, derivatives.size());
    return Eigen::Map<TDerived>{derivatives[index].data()};
  }

  mutable Derivatives derivatives; ///< Derivatives.
  mutable Jacobians jacobians;     ///< Jacobians.
};

} // namespace hyper
