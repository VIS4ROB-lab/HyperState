/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <memory>
#include <set>
// #include <boost/container/flat_set.hpp>
// #include <absl/container/btree_set.h>

#include "hyper/state/interpolators/abstract.hpp"
#include "hyper/state/policies/abstract.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper {

class AbstractState {
 public:
  // Definitions.
  using Range = hyper::Range<Stamp, BoundaryPolicy::LOWER_INCLUSIVE_ONLY>;
  using Parameter = AbstractStamped<Scalar>;
  using Element = std::unique_ptr<Parameter>;

  struct ElementCompare {
    using is_transparent = std::true_type;
    auto operator()(const Element& lhs, const Element& rhs) const -> bool {
      return lhs->stamp() < rhs->stamp();
    }
    auto operator()(const Element& lhs, const Stamp& rhs) const -> bool {
      return lhs->stamp() < rhs;
    }
    auto operator()(const Stamp& lhs, const Element& rhs) const -> bool {
      return lhs < rhs->stamp();
    }
  };

  using Elements = std::set<Element, ElementCompare>;
  // using Elements = boost::container::flat_set<Element, ElementCompare>;
  // using Elements = absl::btree_set<Element, ElementCompare>;

  /// Constructor from interpolator and policy.
  /// \param interpolator Input interpolator.
  /// \param policy Input policy.
  explicit AbstractState(std::unique_ptr<AbstractInterpolator>&& interpolator = nullptr, std::unique_ptr<AbstractPolicy>&& policy = nullptr);

  /// Elements accessor.
  /// \return Elements.
  [[nodiscard]] auto elements() const -> const Elements&;

  /// Elements modifier.
  /// \return Elements.
  auto elements() -> Elements&;

  /// Parameters accessor.
  /// \return Parameters.
  [[nodiscard]] auto parameters() const -> Pointers<Parameter>;

  /// Parameters accessor (stamp-based).
  /// \return Parameters.
  [[nodiscard]] auto parameters(const Stamp& stamp) const -> Pointers<Parameter>;

  /// Evaluates the temporal range.
  /// \return Temporal range.
  [[nodiscard]] auto range() const -> Range;

  /// Interpolator accessor.
  /// \return Interpolator.
  [[nodiscard]] auto interpolator() const -> const std::unique_ptr<AbstractInterpolator>&;

  /// Interpolator modifier.
  /// \return Interpolator.
  [[nodiscard]] auto interpolator() -> std::unique_ptr<AbstractInterpolator>&;

  /// Policy accessor.
  /// \return Policy.
  [[nodiscard]] auto policy() const -> const std::unique_ptr<AbstractPolicy>&;

  /// Policy modifier.
  /// \return Policy.
  [[nodiscard]] auto policy() -> std::unique_ptr<AbstractPolicy>&;

  /// Evaluates the states.
  /// \param state_query State query.
  /// \return Interpolation result.
  [[nodiscard]] auto evaluate(const StateQuery& state_query) const -> StateResult;

  /// Evaluates the states (with external parameters).
  /// \param state_query State query.
  /// \param raw_values Input values.
  /// \return Interpolation result.
  [[nodiscard]] auto evaluate(const StateQuery& state_query, const Scalar* const* raw_values) const -> StateResult;

 private:
  Elements elements_;                                  ///< Elements.
  std::unique_ptr<AbstractInterpolator> interpolator_; ///< Interpolator.
  std::unique_ptr<AbstractPolicy> policy_;             ///< Policy.
};

} // namespace hyper
