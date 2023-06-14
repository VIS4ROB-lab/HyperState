/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <memory>
#include <set>

#include <glog/logging.h>

#include "hyper/matrix.hpp"
#include "hyper/range.hpp"
#include "hyper/state/state.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

template <typename TGroup>
class TemporalState : public State {
 public:
  // Constants.
  static constexpr auto kStatePartitionIndex = 0;
  static constexpr auto kNumPartitions = kStatePartitionIndex + 1;

  // Definitions.
  using Group = TGroup;

  template <typename T>
  using Tangent = variables::Tangent<T>;

  template <typename T>
  using Stamped = variables::Stamped<T>;

  struct Compare {
    using is_transparent = std::true_type;
    auto operator()(const Stamped<TGroup>& lhs, const Stamped<TGroup>& rhs) const -> bool { return lhs.time() < rhs.time(); }
    auto operator()(const Stamped<TGroup>& lhs, const Time& rhs) const -> bool { return lhs.time() < rhs; }
    auto operator()(const Time& lhs, const Stamped<TGroup>& rhs) const -> bool { return lhs < rhs.time(); }
  };

  using StampedParameters = std::set<Stamped<TGroup>, Compare>;

  /// Constructor from uniformity flag and Jacobian type.
  /// \param uniform Uniform.
  /// \param jacobian Jacobian.
  explicit TemporalState(bool uniform, Jacobian jacobian);

  /// Destructor.
  ~TemporalState() override;

  /// Jacobian type accessor.
  /// \return Jacobian type.
  [[nodiscard]] auto jacobian() const -> Jacobian;

  /// Jacobian type setter.
  /// \param jacobian Jacobian.
  auto setJacobian(Jacobian jacobian) -> void;

  /// Retrieves the (ambient) input size.
  /// \return Input size.
  [[nodiscard]] constexpr auto ambientInputSize() const { return Stamped<TGroup>::kNumParameters; }

  /// Retrieves the (ambient) output size.
  /// \return Output size.
  [[nodiscard]] constexpr auto ambientOutputSize() const { return TGroup::kNumParameters; }

  /// Retrieves the local input size.
  /// \return Local input size.
  [[nodiscard]] auto tangentInputSize() const -> int;

  /// Retrieves the local output size.
  /// \return Local output size.
  [[nodiscard]] constexpr auto tangentOutputSize() const { return Tangent<TGroup>::kNumParameters; }

  /// Flag accessor.
  /// \return Flag.
  [[nodiscard]] inline auto isUniform() const -> bool { return uniform_; }

  /// Updates the flag.
  /// \param flag Flag.
  virtual inline auto setUniform(bool flag) -> void { uniform_ = flag; }

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] virtual auto range() const -> InclusiveRange<Scalar> = 0;

  /// Parameters accessor.
  /// \return Parameters.
  auto stampedParameters() const -> const StampedParameters&;

  /// Parameters modifier.
  /// \return Parameters.
  auto stampedParameters() -> StampedParameters&;

  /// Time-based partition accessor.
  /// \param time Query time.
  /// \return Time-based partition.
  [[nodiscard]] virtual auto partition(const Time& time) const -> Partition<Scalar*> = 0;

  /// Time-based parameter blocks accessor.
  /// \return Time-based parameter blocks.
  [[nodiscard]] virtual auto parameterBlocks(const Time& time) const -> std::vector<Scalar*> = 0;

  /// Evaluates this.
  /// \param time Time.
  /// \param derivative Derivative.
  /// \param jacobian Flag.
  /// \param stamped_parameters External pointers.
  /// \return Result.
  virtual auto evaluate(const Time& time, int derivative, bool jacobian = false, const Scalar* const* stamped_parameters = nullptr) const -> Result<TGroup> = 0;  // NOLINT

  /// Publishes this.
  auto publish() -> void final;

 protected:
  bool uniform_;                          ///< Uniform.
  Jacobian jacobian_;                     ///< Jacobian.
  StampedParameters stamped_parameters_;  ///< Stamped parameters.
};

}  // namespace hyper::state
