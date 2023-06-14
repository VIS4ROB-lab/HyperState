/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/state/state.hpp"

namespace hyper::state {

State::State(const String& frame) {
  setFrame(frame);
}
State::~State() = default;

auto State::setFrame(const String& frame) -> void {
  frame_ = frame;
}

auto State::setNode(Node& node, const String& prefix) -> void {
  if (node != node_) {
    // Clear publishers.
    value_publisher_.reset();
    interpolation_publisher_.reset();

    // Create publishers.
    if (node) {
      value_publisher_ = node->create_publisher<geometry_msgs::msg::PoseArray>(prefix + "/value", 1);
      interpolation_publisher_ = node->create_publisher<geometry_msgs::msg::PoseArray>(prefix + "/interpolation", 1);
    }

    node_ = node;
  }
}

}  // namespace hyper::state
