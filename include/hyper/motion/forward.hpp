/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <utility>
#include <vector>

#include <glog/logging.h>

#include "hyper/definitions.hpp"

namespace hyper {

struct StateQuery {
  /// Constructor from query stamp and derivative requests.
  /// \param time Query time.
  /// \param derivative Highest degree of requested derivatives.
  /// \param jacobian Jacobian flag (true if requested).
  StateQuery(const Time& time, Index derivative = 0, const bool jacobian = false) // NOLINT
      : time{time},
        derivative{derivative},
        jacobian{jacobian} {}

  // Members.
  Time time;        ///< Time.
  Index derivative; ///< Highest degree of requested derivatives.
  bool jacobian;    ///< Jacobian flag.
};

struct StateResult {
  // Constants.
  static constexpr auto kValueIndex = 0;
  static constexpr auto kVelocityIndex = 1;
  static constexpr auto kAccelerationIndex = 2;
  static constexpr auto kJerkIndex = 3;

  // Definitions.
  using Derivative = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Jacobian = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
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

  // Members.
  Derivatives derivatives; ///< Derivatives.
  Jacobians jacobians;     ///< Jacobians.
};

struct DerivativeOrder {
  static constexpr auto kValue = 0;
  static constexpr auto kVelocity = 1;
  static constexpr auto kAcceleration = 2;
  static constexpr auto kJerk = 3;
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

template <typename TScalar, typename TDerivative>
struct TemporalMotionQuery {
  TScalar time;
  TDerivative derivative;
  const Scalar* const* values;
  TScalar* const* jacobians;
};

} // namespace hyper
