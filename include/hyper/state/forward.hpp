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

template <typename TOutput, typename TVariable = TOutput>
class TemporalState;

template <typename TOutput>
class DiscreteState;

template <typename TOutput, typename TVariable = TOutput>
class ContinuousState;

template <typename TOutput>
class Result {
 public:
  using Index = Eigen::Index;
  using Scalar = typename TOutput::Scalar;

  using Value = TOutput;
  using Tangent = variables::Tangent<TOutput>;

  Result(const Index& derivative, bool jacobians, const Index& num_inputs, const Index& num_input_parameters)
      : derivative_{derivative}, num_inputs_{num_inputs}, num_input_parameters_{num_input_parameters}, num_parameters_{num_inputs * num_input_parameters} {
    if (!jacobians) {
      matrix.setZero(Tangent::kNumParameters, derivative_);
    } else {
      matrix.setZero(Tangent::kNumParameters, derivative_ + (derivative_ + 1) * num_parameters_);
    }
  }

  inline auto derivative(const Index& k) { return Eigen::Map<Tangent>{matrix.data() + k * Tangent::kNumParameters}; }
  inline auto derivative(const Index& k) const { return Eigen::Map<const Tangent>{matrix.data() + k * Tangent::kNumParameters}; }

  inline auto jacobian(const Index& k) { return matrix.middleCols(derivative_ + k * num_parameters_, num_parameters_); }
  inline auto jacobian(const Index& k) const { return matrix.middleCols(derivative_ + k * num_parameters_, num_parameters_); }
  //inline auto jacobian(const Index& k, const Index& i) { return matrix.middleCols(derivative_ + k * num_parameters_ + i * num_input_parameters_, num_input_parameters_); }
  //inline auto jacobian(const Index& k, const Index& i) const { return matrix.middleCols(derivative_ + k * num_parameters_ + i * num_input_parameters_, num_input_parameters_); }

  template <int NRows, int NCols>
  inline auto jacobian(const Index& k, const Index& i, const Index& start_row, const Index& start_col) {
    return matrix.template block<NRows, NCols>(start_row, derivative_ + k * num_parameters_ + i * num_input_parameters_ + start_col);
  }

  template <int NRows, int NCols>
  inline auto jacobian(const Index& k, const Index& i, const Index& start_row, const Index& start_col) const {
    return matrix.template block<NRows, NCols>(start_row, derivative_ + k * num_parameters_ + i * num_input_parameters_ + start_col);
  }

  Value value;
  MatrixX<Scalar> matrix;

 private:
  Index derivative_;
  Index num_inputs_;
  Index num_input_parameters_;
  Index num_parameters_;
};

}  // namespace hyper::state
