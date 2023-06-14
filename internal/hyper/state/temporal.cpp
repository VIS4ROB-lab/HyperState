/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/state/temporal.hpp"
#include "hyper/variables/se3.hpp"

namespace hyper::state {

using namespace variables;

template <typename TGroup>
TemporalState<TGroup>::TemporalState(bool uniform, Jacobian jacobian) : uniform_{uniform}, jacobian_{jacobian}, stamped_parameters_{} {}

template <typename TGroup>
TemporalState<TGroup>::~TemporalState() = default;

template <typename TGroup>
auto TemporalState<TGroup>::jacobian() const -> Jacobian {
  return jacobian_;
}

template <typename TGroup>
auto TemporalState<TGroup>::setJacobian(Jacobian jacobian) -> void {
  this->jacobian_ = jacobian;
}

template <typename TGroup>
auto TemporalState<TGroup>::tangentInputSize() const -> int {
  switch (this->jacobian_) {
    case Jacobian::TANGENT_TO_TANGENT:
      return Tangent<TGroup>::kNumParameters;
    case Jacobian::TANGENT_TO_STAMPED_TANGENT:
      return Stamped<Tangent<TGroup>>::kNumParameters;
    case Jacobian::TANGENT_TO_GROUP:
      return TGroup::kNumParameters;
    case Jacobian::TANGENT_TO_STAMPED_GROUP:
      return Stamped<TGroup>::kNumParameters;
    default:
      return -1;
  }
}

template <typename TGroup>
auto TemporalState<TGroup>::stampedParameters() const -> const StampedParameters& {
  return stamped_parameters_;
}

template <typename TGroup>
auto TemporalState<TGroup>::stampedParameters() -> StampedParameters& {
  return const_cast<StampedParameters&>(std::as_const(*this).stampedParameters());
}

template <typename TGroup>
auto TemporalState<TGroup>::publish() -> void {
  if (node_) {
    using Pose = geometry_msgs::msg::Pose;
    using Message = geometry_msgs::msg::PoseArray;

    Message message;
    message.header.frame_id = frame_;
    message.header.stamp = node_->now();
    for (const auto& stamped_parameter : stampedParameters()) {
      Pose pose;
      const auto& variable = stamped_parameter.variable();
      if constexpr (std::is_same_v<TGroup, R3>) {
        pose.position.x = variable.x();
        pose.position.y = variable.y();
        pose.position.z = variable.z();
      } else if constexpr (std::is_same_v<TGroup, SU2>) {
        pose.orientation.x = variable.x();
        pose.orientation.y = variable.y();
        pose.orientation.z = variable.z();
        pose.orientation.w = variable.w();
      } else if constexpr (std::is_same_v<TGroup, SE3>) {
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
