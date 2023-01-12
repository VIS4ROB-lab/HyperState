/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <utility>
#include <vector>

#include <glog/logging.h>

#include "hyper/definitions.hpp"
#include "hyper/variables/groups/forward.hpp"
#include "hyper/variables/jacobian.hpp"

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

template <typename TVariable>
struct TemporalMotionResult {
  /// Constructor.
  /// \param k Derivative order.
  /// \param num_inputs Number of inputs.
  /// \param jacobian Jacobian evaluation flag.
  TemporalMotionResult(const Index& k, const Index& num_inputs, bool jacobian = false) : num_inputs_{num_inputs}, num_derivatives_{k + 1} {
    // Allocate memory.
    if (!jacobian) {
      memory_.setZero(TVariable::kNumParameters + (num_derivatives_ - 1) * Tangent<TVariable>::kNumParameters);
    } else {
      memory_.setZero(TVariable::kNumParameters + (num_derivatives_ - 1) * Tangent<TVariable>::kNumParameters +
                      num_derivatives_ * num_inputs_ * Tangent<TVariable>::kNumParameters * Stamped<TVariable>::kNumParameters);
    }

    // Insert value pointer.
    auto data = memory_.data();
    outputs.reserve(num_derivatives_);
    outputs.emplace_back(data);
    data += TVariable::kNumParameters;

    // Insert derivative pointers.
    for (Index i = 0; i < (num_derivatives_ - 1); ++i) {
      outputs.emplace_back(data);
      data += Tangent<TVariable>::kNumParameters;
    }

    // Insert Jacobian pointers.
    if (jacobian) {
      jacobians.reserve(num_derivatives_);
      for (Index i = 0; i < num_derivatives_; ++i) {
        jacobians.emplace_back(data);
        data += num_inputs_ * Tangent<TVariable>::kNumParameters * Stamped<TVariable>::kNumParameters;
      }
    }
  }

  /// Derivative accessor.
  /// \param k Derivative order.
  /// \return k-th derivative.
  auto derivative(const Index& k) const {
    using Output = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    return Eigen::Map<const Output>{outputs[k], (k == 0) ? TVariable::kNumParameters : Tangent<TVariable>::kNumParameters};
  }

  /// Jacobian accessor.
  /// \param k Derivative order.
  /// \return k-th Jacobian.
  auto jacobian(const Index& k) const {
    using Jacobian = Eigen::Matrix<Scalar, Tangent<TVariable>::kNumParameters, Eigen::Dynamic>;
    return Eigen::Map<const Jacobian>{jacobians[k], Tangent<TVariable>::kNumParameters, num_inputs_ * Stamped<TVariable>::kNumParameters};
  }

  /// Value accessor.
  /// \return Value.
  auto value() const {
    return Eigen::Map<const TVariable>{outputs[0]};
  }

  /// Velocity accessor.
  /// \return Velocity.
  auto velocity() const {
    return Eigen::Map<const Tangent<TVariable>>{outputs[1]};
  }

  /// Acceleration accessor.
  /// \return Acceleration.
  auto acceleration() const {
    return Eigen::Map<const Tangent<TVariable>>{outputs[2]};
  }

  Pointers<Scalar> outputs;
  Pointers<Scalar> jacobians;

 private:
  // Definitions.
  using Memory = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  // Members.
  Index num_inputs_;
  Index num_derivatives_;
  Memory memory_;
};

} // namespace hyper
