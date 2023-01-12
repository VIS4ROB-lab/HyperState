/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <memory>
#include <set>

#include <glog/logging.h>

#include "hyper/definitions.hpp"
#include "hyper/motion/motion.hpp"
#include "hyper/range.hpp"
#include "hyper/variables/abstract.hpp"
#include "hyper/variables/jacobian.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper {

template <typename TVariable>
class TemporalMotion : public Motion<typename TVariable::Scalar> {
 public:
  // Definitions.
  using Base = Motion<typename TVariable::Scalar>;
  using Scalar = typename Base::Scalar;

  using Time = Scalar;
  using Range = hyper::Range<Time, BoundaryPolicy::INCLUSIVE>;

  using Element = Stamped<TVariable>;
  using Derivative = typename Base::Derivative;

  // Element compare.
  struct ElementCompare {
    using is_transparent = std::true_type;
    auto operator()(const Element& lhs, const Element& rhs) const -> bool {
      return lhs.stamp() < rhs.stamp();
    }
    auto operator()(const Element& lhs, const Time& rhs) const -> bool {
      return lhs.stamp() < rhs;
    }
    auto operator()(const Time& lhs, const Element& rhs) const -> bool {
      return lhs < rhs.stamp();
    }
  };

  using Elements = std::set<Element, ElementCompare>;

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] virtual auto range() const -> Range = 0;

  /// Elements accessor.
  /// \return Elements.
  auto elements() const -> const Elements& {
    return elements_;
  }

  /// Elements modifier.
  /// \return Elements.
  auto elements() -> Elements& {
    return elements_;
  }

  /// Time-based pointers accessor.
  /// \return Time-based pointers.
  [[nodiscard]] virtual auto pointers(const Time& time) const -> Pointers<Element> = 0;

  /// Evaluates the motion.
  /// \param query Temporal motion query.
  /// \return True on success.

  /// Evaluates this.
  /// \param time Query time.
  /// \param derivative Query derivative.
  /// \param jacobians Jacobians evaluation flag.
  /// \return Result.
  virtual auto evaluate(const Time& time, const Derivative& derivative, bool jacobians) const -> TemporalMotionResult<TVariable> = 0;

  /// Evaluates this.
  /// \param time Query time.
  /// \param derivative Query derivative.
  /// \param jacobians Jacobians evaluation flag.
  /// \param elements Element pointers.
  /// \return Result.
  virtual auto evaluate(const Time& time, const Derivative& derivative, bool jacobians, const Scalar* const* elements) const -> TemporalMotionResult<TVariable> = 0;

 protected:
  Elements elements_; ///< Elements.
};

} // namespace hyper
