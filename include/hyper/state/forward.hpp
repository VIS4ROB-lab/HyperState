/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <utility>
#include <vector>

#include <glog/logging.h>

#include "hyper/variables/groups/forward.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper::state {

template <typename TLabel, typename TVariable>
class LabeledState;

template <typename TOutput, typename TVariable = TOutput>
class TemporalState;

template <typename TOutput>
class DiscreteState;

template <typename TOutput, typename TVariable = TOutput>
class ContinuousState;

enum Derivative {
  VALUE = 0,
  VELOCITY = 1,
  ACCELERATION = 2,
  JERK = 3,
};

enum class JacobianType { TANGENT_TO_TANGENT, TANGENT_TO_MANIFOLD, TANGENT_TO_STAMPED_TANGENT, TANGENT_TO_STAMPED_MANIFOLD };

template <typename TValue>
class Result {
 public:
  // Definitions.
  using Value = TValue;
  using Tangent = variables::Tangent<TValue>;

  // Constants.
  static constexpr auto kNumValueParameters = Value::kNumParameters;
  static constexpr auto kNumTangentParameters = Tangent::kNumParameters;

  Result(int degree, int num_inputs, int num_parameters_per_input, bool has_jacobians = false)
      : value_{},
        storage_{},
        degree_{degree},
        num_inputs_{num_inputs},
        num_parameters_per_input_{num_parameters_per_input},
        num_parameters_{num_inputs * num_parameters_per_input},
        has_jacobians_{has_jacobians} {
    if (has_jacobians) {
      const auto order_ = degree_ + 1;
      storage_.setZero(kNumTangentParameters, degree_ + order_ * num_parameters_);
    } else {
      storage_.setZero(kNumTangentParameters, degree_);
    }
  }

  inline auto degree() const { return degree_; }
  inline auto numInputs() const { return num_inputs_; }
  inline auto numParametersPerInput() const { return num_parameters_per_input_; }
  inline auto numParameters() const { return num_parameters_; }
  inline auto hasJacobians() const { return has_jacobians_; }

  inline auto value() -> Value& { return value_; }
  inline auto value() const -> const Value& { return value_; }
  inline auto velocity() { return tangent(0); }
  inline auto velocity() const { return tangent(0); }
  inline auto acceleration() { return tangent(1); }
  inline auto acceleration() const { return tangent(1); }
  inline auto jerk() { return tangent(2); }
  inline auto jerk() const { return tangent(2); }

  inline auto tangent(int k) { return Eigen::Map<Tangent>{storage_.data() + k * kNumTangentParameters}; }
  inline auto tangent(int k) const { return Eigen::Map<const Tangent>{storage_.data() + k * kNumTangentParameters}; }
  inline auto jacobian(int k) { return storage_.middleCols(degree_ + k * num_parameters_, num_parameters_); }
  inline auto jacobian(int k) const { return storage_.middleCols(degree_ + k * num_parameters_, num_parameters_); }
  inline auto jacobian(int k, int i) { return storage_.middleCols(degree_ + k * num_parameters_ + i * num_parameters_per_input_, num_parameters_per_input_); }
  inline auto jacobian(int k, int i) const { return storage_.middleCols(degree_ + k * num_parameters_ + i * num_parameters_per_input_, num_parameters_per_input_); }

  template <int NRows, int NCols>
  inline auto jacobian(int k, int i, int start_row, int start_col) {
    return storage_.template block<NRows, NCols>(start_row, degree_ + k * num_parameters_ + i * num_parameters_per_input_ + start_col);
  }

  template <int NRows, int NCols>
  inline auto jacobian(int k, int i, int start_row, int start_col) const {
    return storage_.template block<NRows, NCols>(start_row, degree_ + k * num_parameters_ + i * num_parameters_per_input_ + start_col);
  }

  inline auto tangents() { return storage_.leftCols(degree_); }
  inline auto tangents() const { return storage_.leftCols(degree_); }
  inline auto jacobians() { return storage_.rightCols((degree_ + 1) * num_parameters_); }
  inline auto jacobians() const { return storage_.rightCols((degree_ + 1) * num_parameters_); }

 private:
  Value value_;
  MatrixNX<Tangent> storage_;

  int degree_;
  int num_inputs_;
  int num_parameters_per_input_;
  int num_parameters_;
  bool has_jacobians_;
};

}  // namespace hyper::state
