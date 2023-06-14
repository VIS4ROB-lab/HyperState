/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <cmath>
#include <ostream>
#include <type_traits>
#include <vector>

namespace hyper {

template <typename T>
concept Integral = std::is_integral_v<T>;
template <typename T>
concept Float = std::is_floating_point_v<T>;
template <typename T>
concept Real = Integral<T> || Float<T>;

/// Boundary policy enum.
enum class BoundaryPolicy { INCLUSIVE, LOWER_INCLUSIVE_ONLY, UPPER_INCLUSIVE_ONLY, EXCLUSIVE };

template <typename T, BoundaryPolicy boundary_policy>
  requires Integral<T> || Real<T>
class Range {
 public:
  /// Retrieves the boundary policy.
  /// \return Boundary policy.
  [[nodiscard]] constexpr auto policy() const -> BoundaryPolicy { return boundary_policy; }

  /// Returns the lower bound (considering the boundaries).
  /// \return Lower bound of this range.
  auto lowerBound() const -> T;

  /// Returns the upper bound (considering the boundaries).
  /// \return Upper bound of this range.
  auto upperBound() const -> T;

  /// Checks whether a type is smaller than the range.
  /// \param type Query type.
  /// \return True if type compares smaller.
  auto isSmaller(const T& type) const -> bool;

  /// Checks whether a type is greater than the range.
  /// \param type Query type.
  /// \return True if type compares greater.
  auto isGreater(const T& type) const -> bool;

  /// Checks whether value if contained in range.
  /// \param type Query type.
  /// \return True if type is contained.
  inline auto contains(const T& type) const -> bool { return !isSmaller(type) && !isGreater(type); }

  /// Determines the size/length of the range.
  /// \param type Query type.
  /// \return Size of the range (according to boundary policy).
  auto size() const -> T;

  /// Determines if the range is empty.
  /// (i.e. lower bound is larger than upper bound).
  /// \return True if range is empty.
  [[nodiscard]] auto empty() const -> bool { return upperBound() < lowerBound(); }

  /// Samples a range by a sampling rate.
  /// \param rate Sampling rate.
  /// \return Samples.
  auto sample(const T& rate) const -> std::vector<T> {
    // Allocate memory;
    std::vector<T> samples;
    samples.reserve(std::ceil(rate * size()));

    // Fetch lower bound.
    const auto lower_bound = lowerBound();
    const auto upper_bound = upperBound();
    const auto inverse_rate = T{1} / rate;

    // Generate samples.
    auto i = std::size_t{0};
    while (true) {
      const auto sample_i = lower_bound + i * inverse_rate;
      if (sample_i <= upper_bound) {
        samples.emplace_back(sample_i);
        ++i;
      } else {
        return samples;
      }
    }
  }

  /// Retrieves a random sample contained inside the range.
  /// \return Random sample.
  auto sample() const -> T {
    return lowerBound() + T(std::rand()) / T(RAND_MAX) * size();  // NOLINT
  }

  /// Retrieves the closest sample in the range.
  /// \param type Query type.
  /// \return Closest sample.
  [[nodiscard]] auto closest(const T& type) const -> T {
    T closest_sample;
    if (isSmaller(type)) {
      closest_sample = lowerBound();
    } else if (isGreater(type)) {
      closest_sample = upperBound();
    } else {
      closest_sample = type;
    }
    return closest_sample;
  };

  /// Checks whether two ranges intersect.
  /// \tparam OtherType Query type.
  /// \tparam other_boundary_policy Boundary policy of other range.
  /// \param other_range Other range.
  /// \return True if ranges intersect each other.
  template <typename OtherType, BoundaryPolicy other_boundary_policy>
  auto intersects(const Range<OtherType, other_boundary_policy>& other_range) const -> bool {
    return !(upperBound() < other_range.lowerBound() || other_range.upperBound() < lowerBound());
  }

  /// Computes the intersection of ranges.
  /// \tparam OtherType Other range type.
  /// \tparam other_boundary_policy Other boundary policy.
  /// \param other_range Other range.
  /// \return Pair containing boundary values.
  /// \note It is the user's responsibility to correctly handle the boundary policy!
  template <typename OtherType, BoundaryPolicy other_boundary_policy>
  auto intersection(const Range<OtherType, other_boundary_policy>& other_range) const -> std::pair<OtherType, OtherType> {
    const auto lower_value = lowerBound() < other_range.lowerBound() ? other_range.lowerBound() : lowerBound();
    const auto upper_value = upperBound() < other_range.upperBound() ? upperBound() : other_range.upperBound();
    return {lower_value, upper_value};
  }

  /// Stream operator.
  /// \param os Output stream.
  /// \param range Range to output.
  /// \return Modified output stream.
  friend auto operator<<(std::ostream& os, const Range& range) -> std::ostream& {
    if constexpr (boundary_policy == BoundaryPolicy::LOWER_INCLUSIVE_ONLY) {
      return os << "[" << range.lower << ", " << range.upper << ")";
    } else if constexpr (boundary_policy == BoundaryPolicy::UPPER_INCLUSIVE_ONLY) {
      return os << "(" << range.lower << ", " << range.upper << "]";
    } else if constexpr (boundary_policy == BoundaryPolicy::INCLUSIVE) {
      return os << "[" << range.lower << ", " << range.upper << "]";
    } else {
      return os << "(" << range.lower << ", " << range.upper << ")";
    }
  }

  T lower;
  T upper;
};

namespace internal {

template <typename T>
struct BoundaryPolicyEvaluator {
  template <BoundaryPolicy boundary_policy>
  static inline auto LowerBound(const Range<T, boundary_policy>& range) -> T {
    if constexpr (boundary_policy == BoundaryPolicy::LOWER_INCLUSIVE_ONLY || boundary_policy == BoundaryPolicy::INCLUSIVE) {
      return range.lower;
    } else {
      return std::nextafter(range.lower, std::numeric_limits<T>::max());
    }
  }

  template <BoundaryPolicy boundary_policy>
  static inline auto UpperBound(const Range<T, boundary_policy>& range) -> T {
    if constexpr (boundary_policy == BoundaryPolicy::UPPER_INCLUSIVE_ONLY || boundary_policy == BoundaryPolicy::INCLUSIVE) {
      return range.upper;
    } else {
      return std::nextafter(range.upper, -std::numeric_limits<T>::max());
    }
  }

  template <BoundaryPolicy boundary_policy>
  static inline auto IsSmaller(const Range<T, boundary_policy>& range, const T& type) -> bool {
    return type < LowerBound(range);
  }

  template <BoundaryPolicy boundary_policy>
  static inline auto IsGreater(const Range<T, boundary_policy>& range, const T& type) -> bool {
    return UpperBound(range) < type;
  }

  template <BoundaryPolicy boundary_policy>
  static inline auto Size(const Range<T, boundary_policy>& range) -> T {
    return UpperBound(range) - LowerBound(range);
  }
};

}  // namespace internal

template <typename T, BoundaryPolicy boundary_policy>
  requires Integral<T> || Real<T>
auto Range<T, boundary_policy>::lowerBound() const -> T {
  return internal::BoundaryPolicyEvaluator<T>::LowerBound(*this);
}

template <typename T, BoundaryPolicy boundary_policy>
  requires Integral<T> || Real<T>
auto Range<T, boundary_policy>::upperBound() const -> T {
  return internal::BoundaryPolicyEvaluator<T>::UpperBound(*this);
}

template <typename T, BoundaryPolicy boundary_policy>
  requires Integral<T> || Real<T>
auto Range<T, boundary_policy>::isSmaller(const T& type) const -> bool {
  return internal::BoundaryPolicyEvaluator<T>::IsSmaller(*this, type);
}

template <typename T, BoundaryPolicy boundary_policy>
  requires Integral<T> || Real<T>
auto Range<T, boundary_policy>::isGreater(const T& type) const -> bool {
  return internal::BoundaryPolicyEvaluator<T>::IsGreater(*this, type);
}

template <typename T, BoundaryPolicy boundary_policy>
  requires Integral<T> || Real<T>
auto Range<T, boundary_policy>::size() const -> T {
  return internal::BoundaryPolicyEvaluator<T>::Size(*this);
}

template <typename T>
using InclusiveRange = Range<T, BoundaryPolicy::INCLUSIVE>;

template <typename T>
using LowerInclusiveOnlyRange = Range<T, BoundaryPolicy::LOWER_INCLUSIVE_ONLY>;

template <typename T>
using UpperInclusiveOnlyRange = Range<T, BoundaryPolicy::UPPER_INCLUSIVE_ONLY>;

template <typename T>
using ExclusiveRange = Range<T, BoundaryPolicy::EXCLUSIVE>;

}  // namespace hyper
