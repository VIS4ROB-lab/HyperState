/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <utility>
#include <vector>

#include <glog/logging.h>

#include "hyper/variables/groups/forward.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper::state {

enum Derivative : Eigen::Index {
  VALUE = 0,
  VELOCITY = 1,
  ACCELERATION = 2,
  JERK = 3,
};

template <typename TScalar>
class State;

template <typename TLabel, typename TVariable>
class LabeledState;

template <typename TVariable>
class TemporalState;

template <typename TVariable>
class DiscreteState;

template <typename TVariable>
class ContinuousState;

template <typename TOutput>
class Result {
 public:
  using Index = Eigen::Index;
  using Scalar = typename TOutput::Scalar;

  using Value = TOutput;
  using Tangent = variables::Tangent<TOutput>;

  Result(const Index& degree, bool jacobians, const Index& num_inputs, const Index& num_input_parameters)
      : degree_{degree}, num_inputs_{num_inputs}, num_input_parameters_{num_input_parameters}, num_parameters_{num_inputs * num_input_parameters} {
    if (!jacobians) {
      matrix_.setZero(Tangent::kNumParameters, degree_);
    } else {
      matrix_.setZero(Tangent::kNumParameters, degree_ + (degree_ + 1) * num_parameters_);
    }
  }

  inline auto value() -> Value& { return value_; }
  inline auto value() const -> const Value& { return value_; }
  inline auto derivative(const Index& k) { return matrix_.col(k); }
  inline auto derivative(const Index& k) const { return matrix_.col(k); }
  inline auto velocity() { return derivative(0); }
  inline auto velocity() const { return derivative(0); }
  inline auto acceleration() { return derivative(1); }
  inline auto acceleration() const { return derivative(1); }

  inline auto jacobian(const Index& k) { return matrix_.middleCols(degree_ + k * num_parameters_, num_parameters_); }
  inline auto jacobian(const Index& k) const { return matrix_.middleCols(degree_ + k * num_parameters_, num_parameters_); }
  inline auto jacobian(const Index& k, const Index& i) { return matrix_.middleCols(degree_ + k * num_parameters_ + i * num_input_parameters_, num_input_parameters_); }
  inline auto jacobian(const Index& k, const Index& i) const { return matrix_.middleCols(degree_ + k * num_parameters_ + i * num_input_parameters_, num_input_parameters_); }

 private:
  Index degree_;
  Index num_inputs_;
  Index num_input_parameters_;
  Index num_parameters_;

  Value value_;
  MatrixX<Scalar> matrix_;
};

}  // namespace hyper::state
