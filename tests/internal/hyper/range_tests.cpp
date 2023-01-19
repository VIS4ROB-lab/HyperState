/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>
#include <Eigen/Core>

#include "hyper/range.hpp"

namespace hyper::state::tests {

constexpr auto kNumTrials = 10;

using TestTypes = testing::Types<float, double>;

template <typename TScalar>
struct RangeTests : public testing::Test {};

TYPED_TEST_SUITE_P(RangeTests);

TYPED_TEST_P(RangeTests, Empty) {
  for (auto i = 0; i < kNumTrials; ++i) {
    const auto boundary = Eigen::internal::random<TypeParam>();

    const auto inclusive_range = Range<TypeParam, BoundaryPolicy::INCLUSIVE>{boundary, boundary};
    EXPECT_FALSE(inclusive_range.empty());

    const auto lower_inclusive_range = Range<TypeParam, BoundaryPolicy::LOWER_INCLUSIVE_ONLY>{boundary, boundary};
    EXPECT_TRUE(lower_inclusive_range.empty());

    const auto upper_inclusive_range = Range<TypeParam, BoundaryPolicy::UPPER_INCLUSIVE_ONLY>{boundary, boundary};
    EXPECT_TRUE(upper_inclusive_range.empty());

    const auto exclusive_range = Range<TypeParam, BoundaryPolicy::EXCLUSIVE>{boundary, boundary};
    EXPECT_TRUE(exclusive_range.empty());
  }
}

TYPED_TEST_P(RangeTests, LowerBound) {
  for (auto i = 0; i < kNumTrials; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<TypeParam>());
    const auto upper_boundary = std::abs(Eigen::internal::random<TypeParam>());

    const auto inclusive_range = Range<TypeParam, BoundaryPolicy::INCLUSIVE>{lower_boundary, upper_boundary};
    EXPECT_EQ(inclusive_range.lower, inclusive_range.lowerBound());

    const auto lower_inclusive_range = Range<TypeParam, BoundaryPolicy::LOWER_INCLUSIVE_ONLY>{lower_boundary, upper_boundary};
    EXPECT_EQ(lower_inclusive_range.lower, lower_inclusive_range.lowerBound());

    const auto upper_inclusive_range = Range<TypeParam, BoundaryPolicy::UPPER_INCLUSIVE_ONLY>{lower_boundary, upper_boundary};
    EXPECT_LT(upper_inclusive_range.lower, upper_inclusive_range.lowerBound());

    const auto exclusive_range = Range<TypeParam, BoundaryPolicy::EXCLUSIVE>{lower_boundary, upper_boundary};
    EXPECT_LT(exclusive_range.lower, exclusive_range.lowerBound());
  }
}

TYPED_TEST_P(RangeTests, UpperBound) {
  for (auto i = 0; i < kNumTrials; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<TypeParam>());
    const auto upper_boundary = std::abs(Eigen::internal::random<TypeParam>());

    const auto inclusive_range = Range<TypeParam, BoundaryPolicy::INCLUSIVE>{lower_boundary, upper_boundary};
    EXPECT_EQ(inclusive_range.upper, inclusive_range.upperBound());

    const auto lower_inclusive_range = Range<TypeParam, BoundaryPolicy::LOWER_INCLUSIVE_ONLY>{lower_boundary, upper_boundary};
    EXPECT_GT(lower_inclusive_range.upper, lower_inclusive_range.upperBound());

    const auto upper_inclusive_range = Range<TypeParam, BoundaryPolicy::UPPER_INCLUSIVE_ONLY>{lower_boundary, upper_boundary};
    EXPECT_EQ(upper_inclusive_range.upper, upper_inclusive_range.upperBound());

    const auto exclusive_range = Range<TypeParam, BoundaryPolicy::EXCLUSIVE>{lower_boundary, upper_boundary};
    EXPECT_GT(exclusive_range.upper, exclusive_range.upperBound());
  }
}

TYPED_TEST_P(RangeTests, IsSmaller) {
  for (auto i = 0; i < kNumTrials; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<TypeParam>());
    const auto upper_boundary = std::abs(Eigen::internal::random<TypeParam>());
    const auto smaller_element = std::nextafter(lower_boundary, -std::numeric_limits<TypeParam>::max());

    const auto inclusive_range = Range<TypeParam, BoundaryPolicy::INCLUSIVE>{lower_boundary, upper_boundary};
    EXPECT_FALSE(inclusive_range.isSmaller(lower_boundary));
    EXPECT_TRUE(inclusive_range.isSmaller(smaller_element));

    const auto lower_inclusive_range = Range<TypeParam, BoundaryPolicy::LOWER_INCLUSIVE_ONLY>{lower_boundary, upper_boundary};
    EXPECT_FALSE(lower_inclusive_range.isSmaller(lower_boundary));
    EXPECT_TRUE(lower_inclusive_range.isSmaller(smaller_element));

    const auto upper_inclusive_range = Range<TypeParam, BoundaryPolicy::UPPER_INCLUSIVE_ONLY>{lower_boundary, upper_boundary};
    EXPECT_TRUE(upper_inclusive_range.isSmaller(lower_boundary));

    const auto exclusive_range = Range<TypeParam, BoundaryPolicy::EXCLUSIVE>{lower_boundary, upper_boundary};
    EXPECT_TRUE(exclusive_range.isSmaller(lower_boundary));
  }
}

TYPED_TEST_P(RangeTests, IsGreater) {
  for (auto i = 0; i < kNumTrials; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<TypeParam>());
    const auto upper_boundary = std::abs(Eigen::internal::random<TypeParam>());
    const auto larger_element = std::nextafter(upper_boundary, std::numeric_limits<TypeParam>::max());

    const auto inclusive_range = Range<TypeParam, BoundaryPolicy::INCLUSIVE>{lower_boundary, upper_boundary};
    EXPECT_FALSE(inclusive_range.isGreater(upper_boundary));
    EXPECT_TRUE(inclusive_range.isGreater(larger_element));

    const auto lower_inclusive_range = Range<TypeParam, BoundaryPolicy::LOWER_INCLUSIVE_ONLY>{lower_boundary, upper_boundary};
    EXPECT_TRUE(lower_inclusive_range.isGreater(upper_boundary));

    const auto upper_inclusive_range = Range<TypeParam, BoundaryPolicy::UPPER_INCLUSIVE_ONLY>{lower_boundary, upper_boundary};
    EXPECT_FALSE(upper_inclusive_range.isGreater(upper_boundary));
    EXPECT_TRUE(lower_inclusive_range.isGreater(larger_element));

    const auto exclusive_range = Range<TypeParam, BoundaryPolicy::EXCLUSIVE>{lower_boundary, upper_boundary};
    EXPECT_TRUE(exclusive_range.isGreater(upper_boundary));
  }
}

TYPED_TEST_P(RangeTests, Contains) {
  for (auto i = 0; i < kNumTrials; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<TypeParam>());
    const auto upper_boundary = std::abs(Eigen::internal::random<TypeParam>());
    const auto middle = 0.5 * (upper_boundary + lower_boundary);

    const auto inclusive_range = Range<TypeParam, BoundaryPolicy::INCLUSIVE>{lower_boundary, upper_boundary};
    const auto lower_inclusive_range = Range<TypeParam, BoundaryPolicy::LOWER_INCLUSIVE_ONLY>{lower_boundary, upper_boundary};
    const auto upper_inclusive_range = Range<TypeParam, BoundaryPolicy::UPPER_INCLUSIVE_ONLY>{lower_boundary, upper_boundary};
    const auto exclusive_range = Range<TypeParam, BoundaryPolicy::EXCLUSIVE>{lower_boundary, upper_boundary};

    EXPECT_TRUE(inclusive_range.contains(lower_boundary));
    EXPECT_TRUE(lower_inclusive_range.contains(lower_boundary));
    EXPECT_FALSE(upper_inclusive_range.contains(lower_boundary));
    EXPECT_FALSE(exclusive_range.contains(lower_boundary));

    EXPECT_TRUE(inclusive_range.contains(middle));
    EXPECT_TRUE(lower_inclusive_range.contains(middle));
    EXPECT_TRUE(upper_inclusive_range.contains(middle));
    EXPECT_TRUE(exclusive_range.contains(middle));

    EXPECT_TRUE(inclusive_range.contains(upper_boundary));
    EXPECT_FALSE(lower_inclusive_range.contains(upper_boundary));
    EXPECT_TRUE(upper_inclusive_range.contains(upper_boundary));
    EXPECT_FALSE(exclusive_range.contains(upper_boundary));
  }
}

TYPED_TEST_P(RangeTests, Sample) {
  for (auto i = 0; i < kNumTrials; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<TypeParam>());
    const auto upper_boundary = std::abs(Eigen::internal::random<TypeParam>());

    const auto inclusive_range = Range<TypeParam, BoundaryPolicy::INCLUSIVE>{lower_boundary, upper_boundary};

    constexpr auto kNumberOfSamples = 100;
    for (auto j = 0; j < kNumberOfSamples; ++j) {
      EXPECT_TRUE(inclusive_range.contains(inclusive_range.sample()));
    }
  }
}

TYPED_TEST_P(RangeTests, Closest) {
  for (auto i = 0; i < kNumTrials; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<TypeParam>());
    const auto upper_boundary = std::abs(Eigen::internal::random<TypeParam>());
    const auto smaller_element = std::nextafter(lower_boundary, -std::numeric_limits<TypeParam>::max());
    const auto larger_element = std::nextafter(upper_boundary, std::numeric_limits<TypeParam>::max());

    const auto inclusive_range = Range<TypeParam, BoundaryPolicy::INCLUSIVE>{lower_boundary, upper_boundary};

    auto closest_sample = inclusive_range.closest(smaller_element);
    EXPECT_TRUE(inclusive_range.contains(closest_sample));
    EXPECT_EQ(closest_sample, inclusive_range.lowerBound());

    const auto sample = inclusive_range.sample();
    closest_sample = inclusive_range.closest(sample);
    EXPECT_TRUE(inclusive_range.contains(closest_sample));
    EXPECT_EQ(closest_sample, sample);

    closest_sample = inclusive_range.closest(larger_element);
    EXPECT_TRUE(inclusive_range.contains(closest_sample));
    EXPECT_EQ(closest_sample, inclusive_range.upperBound());
  }
}

REGISTER_TYPED_TEST_SUITE_P(RangeTests, Empty, LowerBound, UpperBound, IsSmaller, IsGreater, Contains, Sample, Closest);
INSTANTIATE_TYPED_TEST_SUITE_P(HyperTests, RangeTests, TestTypes);

}  // namespace hyper::state::tests
