/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/state/temporal.hpp"
#include "hyper/variables/se3.hpp"

namespace hyper::state {

using namespace variables;

template <typename TElement>
TemporalState<TElement>::TemporalState(bool is_uniform, JacobianType jacobian_type) : is_uniform_{is_uniform}, jacobian_type_{jacobian_type}, stamped_elements_{} {}

template <typename TElement>
TemporalState<TElement>::~TemporalState() = default;

template <typename TElement>
auto TemporalState<TElement>::jacobianType() const -> JacobianType {
  return jacobian_type_;
}

template <typename TElement>
auto TemporalState<TElement>::setJacobianType(JacobianType jacobian_type) -> void {
  this->jacobian_type_ = jacobian_type;
}

template <typename TElement>
auto TemporalState<TElement>::tangentInputSize() const -> int {
  switch (this->jacobian_type_) {
    case JacobianType::TANGENT_TO_TANGENT:
      return ElementTangent::kNumParameters;
    case JacobianType::TANGENT_TO_STAMPED_TANGENT:
      return StampedElementTangent::kNumParameters;
    case JacobianType::TANGENT_TO_MANIFOLD:
      return Element::kNumParameters;
    case JacobianType::TANGENT_TO_STAMPED_MANIFOLD:
      return StampedElement::kNumParameters;
    default:
      return -1;
  }
}

template <typename TElement>
auto TemporalState<TElement>::stampedElements() const -> const StampedElements& {
  return stamped_elements_;
}

template <typename TElement>
auto TemporalState<TElement>::stampedElements() -> StampedElements& {
  return const_cast<StampedElements&>(std::as_const(*this).stampedElements());
}

template <typename TElement>
auto TemporalState<TElement>::publish() -> void {
  if (node_) {
    using Pose = geometry_msgs::msg::Pose;
    using Message = geometry_msgs::msg::PoseArray;

    Message message;
    message.header.frame_id = frame_;
    message.header.stamp = node_->now();
    for (const auto& stamped_element : stampedElements()) {
      Pose pose;
      const auto& variable = stamped_element.variable();
      if constexpr (std::is_same_v<TElement, R3>) {
        pose.position.x = variable.x();
        pose.position.y = variable.y();
        pose.position.z = variable.z();
      } else if constexpr (std::is_same_v<TElement, SU2>) {
        pose.orientation.x = variable.x();
        pose.orientation.y = variable.y();
        pose.orientation.z = variable.z();
        pose.orientation.w = variable.w();
      } else if constexpr (std::is_same_v<TElement, SE3>) {
        const auto& rotation = variable.rotation();
        const auto& translation = variable.translation();
        pose.position.x = translation.x();
        pose.position.y = translation.y();
        pose.position.z = translation.z();
        pose.orientation.x = rotation.x();
        pose.orientation.y = rotation.y();
        pose.orientation.z = rotation.z();
        pose.orientation.w = rotation.w();
      }
      message.poses.emplace_back(pose);
    }
    value_publisher_->publish(message);
  }
}

template class TemporalState<R3>;
template class TemporalState<SU2>;
template class TemporalState<SE3>;

}  // namespace hyper::state
