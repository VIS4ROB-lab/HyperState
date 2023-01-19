/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <memory>
#include <set>

#include <glog/logging.h>

#include "hyper/state/state.hpp"
#include "hyper/variables/jacobian.hpp"
#include "hyper/variables/stamped.hpp"
#include "hyper/variables/variable.hpp"
#include "range.hpp"

namespace hyper::state {

template <typename TVariable>
class TemporalState : public State<typename TVariable::Scalar> {
 public:
  // Definitions.
  using Base = State<typename TVariable::Scalar>;

  using Index = typename Base::Index;
  using Scalar = typename Base::Scalar;

  using Time = Scalar;
  using Range = state::Range<Time, BoundaryPolicy::INCLUSIVE>;

  using Element = variables::Stamped<TVariable>;

  // Element compare.
  struct ElementCompare {
    using is_transparent = std::true_type;
    auto operator()(const Element& lhs, const Element& rhs) const -> bool { return lhs.stamp() < rhs.stamp(); }
    auto operator()(const Element& lhs, const Time& rhs) const -> bool { return lhs.stamp() < rhs; }
    auto operator()(const Time& lhs, const Element& rhs) const -> bool { return lhs < rhs.stamp(); }
  };

  using Elements = std::set<Element, ElementCompare>;

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] virtual auto range() const -> Range = 0;

  /// Elements accessor.
  /// \return Elements.
  auto elements() const -> const Elements& { return elements_; }

  /// Elements modifier.
  /// \return Elements.
  auto elements() -> Elements& { return elements_; }

  /// Time-based pointers accessor.
  /// \return Time-based pointers.
  [[nodiscard]] virtual auto pointers(const Time& time) const -> std::vector<Element*> = 0;

  /// Evaluates this.
  /// \param time Query time.
  /// \param derivative Query derivative.
  /// \param jacobians Jacobians evaluation flag.
  /// \return Result.
  virtual auto evaluate(const Time& time, const Index& derivative, bool jacobians) const -> Result<TVariable> = 0;

  /// Evaluates this.
  /// \param time Query time.
  /// \param derivative Query derivative.
  /// \param jacobians Jacobians evaluation flag.
  /// \param elements Element pointers.
  /// \return Result.
  virtual auto evaluate(const Time& time, const Index& derivative, bool jacobians, const Scalar* const* elements) const -> Result<TVariable> = 0;

 protected:
  Elements elements_;  ///< Elements.
};

}  // namespace hyper::state
