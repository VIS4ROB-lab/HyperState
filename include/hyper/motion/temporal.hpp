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
#include "hyper/variables/definitions/jacobian.hpp"
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
  using Query = TemporalMotionQuery<Scalar>;

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
  virtual auto evaluate(const Query& query) const -> bool = 0;

  /// Evaluates the motion.
  /// \param query Temporal motion query.
  /// \param inputs Input pointers.
  /// \return True on success.
  virtual auto evaluate(const Query& query, const Scalar* const* inputs) const -> bool = 0;

 protected:
  /// Extracts the timestamps.
  /// \param pointers Input pointers.
  /// \param num_pointers Number of pointers.
  /// \return Timestamps.
  auto extractTimestamps(const Scalar* const* pointers, const Index& num_pointers) const -> std::vector<Scalar> {
    std::vector<Scalar> timestamps;
    timestamps.reserve(num_pointers);
    for (auto i = Index{0}; i < num_pointers; ++i) {
      const auto element_i = Eigen::Map<const Element>{pointers[i]};
      timestamps.emplace_back(element_i.stamp());
    }
    return timestamps;
  }

  Elements elements_; ///< Elements.
};

} // namespace hyper
