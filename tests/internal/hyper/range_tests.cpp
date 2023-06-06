/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>
#include <Eigen/Core>

#include "hyper/definitions.hpp"
#include "hyper/state/range.hpp"

namespace hyper::state::tests {

constexpr auto kItr = 10;

TEST(RangeTests, Empty) {
  for (auto i = 0; i < kItr; ++i) {
    const auto boundary = Eigen::internal::random<Scalar>();

    const auto inclusive_range = InclusiveRange<Scalar>{boundary, boundary};
    EXPECT_FALSE(inclusive_range.empty());

    const auto lower_inclusive_range = LowerInclusiveOnlyRange<Scalar>{boundary, boundary};
    EXPECT_TRUE(lower_inclusive_range.empty());

    const auto upper_inclusive_range = UpperInclusiveOnlyRange<Scalar>{boundary, boundary};
    EXPECT_TRUE(upper_inclusive_range.empty());

    const auto exclusive_range = ExclusiveRange<Scalar>{boundary, boundary};
    EXPECT_TRUE(exclusive_range.empty());
  }
}

TEST(RangeTests, LowerBound) {
  for (auto i = 0; i < kItr; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<Scalar>());
    const auto upper_boundary = std::abs(Eigen::internal::random<Scalar>());

    const auto inclusive_range = InclusiveRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_EQ(inclusive_range.lower, inclusive_range.lowerBound());

    const auto lower_inclusive_range = LowerInclusiveOnlyRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_EQ(lower_inclusive_range.lower, lower_inclusive_range.lowerBound());

    const auto upper_inclusive_range = UpperInclusiveOnlyRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_LT(upper_inclusive_range.lower, upper_inclusive_range.lowerBound());

    const auto exclusive_range = ExclusiveRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_LT(exclusive_range.lower, exclusive_range.lowerBound());
  }
}

TEST(RangeTests, UpperBound) {
  for (auto i = 0; i < kItr; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<Scalar>());
    const auto upper_boundary = std::abs(Eigen::internal::random<Scalar>());

    const auto inclusive_range = InclusiveRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_EQ(inclusive_range.upper, inclusive_range.upperBound());

    const auto lower_inclusive_range = LowerInclusiveOnlyRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_GT(lower_inclusive_range.upper, lower_inclusive_range.upperBound());

    const auto upper_inclusive_range = UpperInclusiveOnlyRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_EQ(upper_inclusive_range.upper, upper_inclusive_range.upperBound());

    const auto exclusive_range = ExclusiveRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_GT(exclusive_range.upper, exclusive_range.upperBound());
  }
}

TEST(RangeTests, IsSmaller) {
  for (auto i = 0; i < kItr; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<Scalar>());
    const auto upper_boundary = std::abs(Eigen::internal::random<Scalar>());
    const auto smaller_element = std::nextafter(lower_boundary, -std::numeric_limits<Scalar>::max());

    const auto inclusive_range = InclusiveRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_FALSE(inclusive_range.isSmaller(lower_boundary));
    EXPECT_TRUE(inclusive_range.isSmaller(smaller_element));

    const auto lower_inclusive_range = LowerInclusiveOnlyRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_FALSE(lower_inclusive_range.isSmaller(lower_boundary));
    EXPECT_TRUE(lower_inclusive_range.isSmaller(smaller_element));

    const auto upper_inclusive_range = UpperInclusiveOnlyRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_TRUE(upper_inclusive_range.isSmaller(lower_boundary));

    const auto exclusive_range = ExclusiveRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_TRUE(exclusive_range.isSmaller(lower_boundary));
  }
}

TEST(RangeTests, IsGreater) {
  for (auto i = 0; i < kItr; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<Scalar>());
    const auto upper_boundary = std::abs(Eigen::internal::random<Scalar>());
    const auto larger_element = std::nextafter(upper_boundary, std::numeric_limits<Scalar>::max());

    const auto inclusive_range = InclusiveRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_FALSE(inclusive_range.isGreater(upper_boundary));
    EXPECT_TRUE(inclusive_range.isGreater(larger_element));

    const auto lower_inclusive_range = LowerInclusiveOnlyRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_TRUE(lower_inclusive_range.isGreater(upper_boundary));

    const auto upper_inclusive_range = UpperInclusiveOnlyRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_FALSE(upper_inclusive_range.isGreater(upper_boundary));
    EXPECT_TRUE(lower_inclusive_range.isGreater(larger_element));

    const auto exclusive_range = ExclusiveRange<Scalar>{lower_boundary, upper_boundary};
    EXPECT_TRUE(exclusive_range.isGreater(upper_boundary));
  }
}

TEST(RangeTests, Contains) {
  for (auto i = 0; i < kItr; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<Scalar>());
    const auto upper_boundary = std::abs(Eigen::internal::random<Scalar>());
    const auto middle = 0.5 * (upper_boundary + lower_boundary);

    const auto inclusive_range = InclusiveRange<Scalar>{lower_boundary, upper_boundary};
    const auto lower_inclusive_range = LowerInclusiveOnlyRange<Scalar>{lower_boundary, upper_boundary};
    const auto upper_inclusive_range = UpperInclusiveOnlyRange<Scalar>{lower_boundary, upper_boundary};
    const auto exclusive_range = ExclusiveRange<Scalar>{lower_boundary, upper_boundary};

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

TEST(RangeTests, Sample) {
  for (auto i = 0; i < kItr; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<Scalar>());
    const auto upper_boundary = std::abs(Eigen::internal::random<Scalar>());

    const auto inclusive_range = InclusiveRange<Scalar>{lower_boundary, upper_boundary};

    constexpr auto kNumberOfSamples = 100;
    for (auto j = 0; j < kNumberOfSamples; ++j) {
      EXPECT_TRUE(inclusive_range.contains(inclusive_range.sample()));
    }
  }
}

TEST(RangeTests, Closest) {
  for (auto i = 0; i < kItr; ++i) {
    const auto lower_boundary = -std::abs(Eigen::internal::random<Scalar>());
    const auto upper_boundary = std::abs(Eigen::internal::random<Scalar>());
    const auto smaller_element = std::nextafter(lower_boundary, -std::numeric_limits<Scalar>::max());
    const auto larger_element = std::nextafter(upper_boundary, std::numeric_limits<Scalar>::max());

    const auto inclusive_range = InclusiveRange<Scalar>{lower_boundary, upper_boundary};

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

}  // namespace hyper::state::tests
