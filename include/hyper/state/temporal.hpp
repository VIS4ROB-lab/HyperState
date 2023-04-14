/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <memory>
#include <set>

#include <glog/logging.h>

#include "hyper/state/forward.hpp"

#include "hyper/jacobian.hpp"
#include "hyper/state/range.hpp"
#include "hyper/variables/stamped.hpp"
#include "hyper/variables/variable.hpp"

namespace hyper::state {

template <typename TElement>
class TemporalState {
 public:
  // Constants.
  static constexpr auto kStatePartitionIndex = 0;
  static constexpr auto kNumPartitions = kStatePartitionIndex + 1;

  static constexpr auto kDefaultJacobianType = JacobianType::TANGENT_TO_STAMPED_MANIFOLD;

  // Definitions.
  using Time = typename TElement::Scalar;
  using Scalar = typename TElement::Scalar;
  using Range = state::Range<Time, BoundaryPolicy::INCLUSIVE>;

  using Element = TElement;
  using ElementTangent = variables::Tangent<TElement>;
  using StampedElement = variables::Stamped<TElement>;
  using StampedElementTangent = variables::Stamped<ElementTangent>;

  // Stamped element compare.
  struct StampedElementCompare {
    using is_transparent = std::true_type;
    auto operator()(const StampedElement& lhs, const StampedElement& rhs) const -> bool { return lhs.time() < rhs.time(); }
    auto operator()(const StampedElement& lhs, const Time& rhs) const -> bool { return lhs.time() < rhs; }
    auto operator()(const Time& lhs, const StampedElement& rhs) const -> bool { return lhs < rhs.time(); }
  };

  using StampedElements = std::set<StampedElement, StampedElementCompare>;

  /// Constructor from uniformity flag and Jacobian type.
  /// \param is_uniform Uniformity flag.
  /// \param jacobian_type Jacobian type.
  explicit TemporalState(bool is_uniform, JacobianType jacobian_type) : is_uniform_{is_uniform}, jacobian_type_{jacobian_type}, stamped_elements_{} {}

  /// Destructor.
  virtual ~TemporalState() = default;

  /// Jacobian type accessor.
  /// \return Jacobian type.
  [[nodiscard]] inline auto jacobianType() const -> JacobianType { return jacobian_type_; }

  /// Jacobian type setter.
  /// \param jacobian_type Jacobian type.
  inline auto setJacobianType(JacobianType jacobian_type) -> void { this->jacobian_type_ = jacobian_type; }

  /// Retrieves the (ambient) input size.
  /// \return Input size.
  [[nodiscard]] constexpr auto ambientInputSize() const { return StampedElement::kNumParameters; }

  /// Retrieves the (ambient) output size.
  /// \return Output size.
  [[nodiscard]] constexpr auto ambientOutputSize() const { return Element::kNumParameters; }

  /// Retrieves the local input size.
  /// \return Local input size.
  [[nodiscard]] inline auto tangentInputSize() const {
    switch (this->jacobian_type_) {
      case JacobianType::TANGENT_TO_TANGENT:
        return ElementTangent::kNumParameters;
      case JacobianType::TANGENT_TO_STAMPED_TANGENT:
        return StampedElementTangent::kNumParameters;
      case JacobianType::TANGENT_TO_MANIFOLD:
        return Element::kNumParameters;
      case JacobianType::TANGENT_TO_STAMPED_MANIFOLD:
        return StampedElement::kNumParameters;
      default: {
        LOG(FATAL) << "Unknown Jacobian type.";
        return -1;
      }
    }
  }

  /// Retrieves the local output size.
  /// \return Local output size.
  [[nodiscard]] constexpr auto tangentOutputSize() const { return ElementTangent::kNumParameters; }

  /// Flag accessor.
  /// \return Flag.
  [[nodiscard]] inline auto isUniform() const -> bool { return is_uniform_; }

  /// Updates the flag.
  /// \param flag Flag.
  virtual inline auto setUniform(bool flag) -> void { is_uniform_ = flag; }

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] virtual auto range() const -> Range = 0;

  /// Elements accessor.
  /// \return Elements.
  inline auto stampedElements() const -> const StampedElements& { return stamped_elements_; }

  /// Elements modifier.
  /// \return Elements.
  inline auto stampedElements() -> StampedElements& { return const_cast<StampedElements&>(std::as_const(*this).stampedElements()); }

  /// Time-based partition accessor.
  /// \param time Query time.
  /// \return Time-based partition.
  [[nodiscard]] virtual auto partition(const Time& time) const -> variables::Partition<Scalar*> = 0;

  /// Time-based parameter blocks accessor.
  /// \return Time-based parameter blocks.
  [[nodiscard]] virtual auto parameterBlocks(const Time& time) const -> std::vector<Scalar*> = 0;

  /// Evaluates this.
  /// \param time Time.
  /// \param derivative Derivative.
  /// \param jacobian Flag.
  /// \param stamped_elements External pointers.
  /// \return Result.
  virtual auto evaluate(const Time& time, int derivative, bool jacobian = false, const Scalar* const* stamped_elements = nullptr) const -> Result<TElement> = 0;  // NOLINT

 protected:
  bool is_uniform_;                   ///< Uniformity flag.
  JacobianType jacobian_type_;        ///< Jacobian type.
  StampedElements stamped_elements_;  ///< Stamped elements.
};

}  // namespace hyper::state
