/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <vector>

#include <Eigen/Core>

#include "hyper/range.hpp"

namespace hyper {

using Scalar = double;
using Identifier = std::size_t;

using Index = Eigen::Index;
using IndexRange = Range<Index, BoundaryPolicy::LOWER_INCLUSIVE_ONLY>;

using Stamp = Scalar;
using Stamps = std::vector<Stamp>;

template <typename T>
using Pointers = std::vector<T*>;

} // namespace hyper
