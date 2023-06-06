/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <iostream>

#include <glog/logging.h>

#include "hyper/matrix.hpp"
#include "hyper/state/continuous.hpp"
#include "hyper/variables/se3.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

using namespace variables;

template <typename TElement>
ContinuousState<TElement>::ContinuousState(std::unique_ptr<TemporalInterpolator>&& interpolator, bool is_uniform, const JacobianType jacobian_type)
    : Base{is_uniform, jacobian_type}, layout_{}, interpolator_{} {
  swapInterpolator(interpolator);
}

template <typename TElement>
auto ContinuousState<TElement>::setUniform(bool flag) -> void {
  Base::setUniform(flag);
  layout_ = interpolator_->layout(flag);
}

template <typename TElement>
auto ContinuousState<TElement>::range() const -> Range {
  DCHECK_LE(layout_.outer_size, this->stamped_elements_.size());
  const auto v0_itr = std::next(this->stamped_elements_.cbegin(), layout_.left_margin - 1);
  const auto vn_itr = std::next(this->stamped_elements_.crbegin(), layout_.right_margin - 1);
  const auto t0 = v0_itr->time();
  const auto tn = vn_itr->time();
  DCHECK_LT(t0, tn);
  return {t0, std::nexttoward(tn, t0)};
}

template <typename TElement>
auto ContinuousState<TElement>::partition(const Time& time) const -> Partition<Scalar*> {
  const auto [begin, end, num_stamped_elements] = iterators(time);
  Partition<Scalar*> partition;
  partition.reserve(num_stamped_elements);
  const auto input_size = this->tangentInputSize();
  std::transform(begin, end, std::back_inserter(partition), [&input_size](const auto& stamped_element) -> Partition<Scalar*>::value_type {
    return {const_cast<Scalar*>(stamped_element.data()), input_size};
  });
  return partition;
}

template <typename TElement>
auto ContinuousState<TElement>::parameterBlocks(const Time& time) const -> std::vector<Scalar*> {
  const auto [begin, end, num_variables] = iterators(time);
  std::vector<Scalar*> ptrs;
  ptrs.reserve(num_variables);
  std::transform(begin, end, std::back_inserter(ptrs), [](const auto& element) { return const_cast<Scalar*>(element.data()); });
  return ptrs;
}

template <typename TElement>
auto ContinuousState<TElement>::interpolator() const -> const TemporalInterpolator& {
  return *interpolator_;
}

template <typename TElement>
auto ContinuousState<TElement>::swapInterpolator(std::unique_ptr<TemporalInterpolator>& interpolator) -> void {
  CHECK(interpolator != nullptr);
  layout_ = interpolator->layout(this->is_uniform_);
  interpolator_.swap(interpolator);
}

template <typename TElement>
auto ContinuousState<TElement>::layout() const -> const TemporalInterpolatorLayout& {
  return layout_;
}

template <typename TElement>
auto ContinuousState<TElement>::evaluate(const Time& time, int derivative, bool jacobian, const Scalar* const* stamped_elements) const -> Result<TElement> {
  if (!stamped_elements) {
    const auto [begin, end, num_elements] = iterators(time);
    std::vector<const Scalar*> ptrs;
    ptrs.reserve(num_elements);
    std::transform(begin, end, std::back_inserter(ptrs), [](const auto& element) { return element.data(); });
    DCHECK_EQ(ptrs.size(), num_elements);
    return evaluate(time, derivative, jacobian, ptrs.data());

  } else {
    // Constants.
    constexpr auto kStampOffset = StampedElement::kStampOffset;
    constexpr auto kVariableOffset = StampedElement::kVariableOffset;

    // Fetch layout.
    const auto s_idx = layout_.left_padding;
    const auto e_idx = layout_.left_padding + layout_.inner_size - 1;

    // Compute normalized time.
    const auto idx = layout_.left_margin - 1;
    const auto dt = time - stamped_elements[idx][kStampOffset];
    const auto i_dt = Scalar{1} / (stamped_elements[idx + 1][kStampOffset] - stamped_elements[idx][kStampOffset]);
    const auto ut = dt * i_dt;

    // Evaluate output.
    auto result = Result<TElement>{derivative, layout_.outer_size, this->tangentInputSize(), jacobian};
    const auto weights = interpolator_->evaluate(ut, i_dt, derivative, !this->is_uniform_ ? stamped_elements : nullptr, kStampOffset);
    SpatialInterpolator<TElement>::evaluate(result, weights.data(), stamped_elements, s_idx, e_idx, kVariableOffset);

    // Convert Jacobians.
    if (jacobian && (this->jacobian_type_ == JacobianType::TANGENT_TO_MANIFOLD || this->jacobian_type_ == JacobianType::TANGENT_TO_STAMPED_MANIFOLD)) {
      for (auto i = s_idx; i <= e_idx; ++i) {
        const auto J_a = Eigen::Map<const Element>{stamped_elements[i] + kVariableOffset}.tMinusJacobian();
        for (auto k = 0; k <= derivative; ++k) {
          result.template jacobian<ElementTangent::kNumParameters, Element::kNumParameters>(k, i, 0, kVariableOffset) =
              result.template jacobian<ElementTangent::kNumParameters, ElementTangent::kNumParameters>(k, i, 0, kVariableOffset) * J_a;
        }
      }
    }

    return result;
  }
}

template <typename TElement>
auto ContinuousState<TElement>::iterators(const Time& time) const -> std::tuple<Iterator, Iterator, int> {
  DCHECK(range().contains(time)) << "Range does not contain time.";
  DCHECK_LE(layout_.outer_size, this->stamped_elements_.size());
  const auto itr = this->stamped_elements_.upper_bound(time);
  const auto begin = std::prev(itr, layout_.left_margin);
  const auto end = std::next(itr, layout_.right_margin);
  return {begin, end, layout_.outer_size};
}

template class ContinuousState<R3>;
template class ContinuousState<SU2>;
template class ContinuousState<SE3>;

}  // namespace hyper::state
