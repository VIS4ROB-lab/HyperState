/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <geometry_msgs/msg/pose_array.hpp>
#include <rclcpp/node.hpp>

#include "hyper/state/forward.hpp"

#include "hyper/definitions.hpp"

namespace hyper::state {

class State {
 public:
  // Definitions.
  using Node = rclcpp::Node::SharedPtr;

  /// Default constructor.
  explicit State(const String& frame = "map");

  /// Default destructor.
  virtual ~State();

  /// Sets the frame.
  /// \param frame Frame.
  auto setFrame(const String& frame) -> void;

  /// Sets the node.
  /// \param node Node.
  auto setNode(Node& node, const String& prefix = "hyper/state") -> void;

  /// Publishes this.
  virtual auto publish() -> void = 0;

 protected:
  // Definitions.
  using Publisher = rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr;

  Publisher value_publisher_;          ///< Value publisher.
  Publisher interpolation_publisher_;  ///< Interpolation publisher.

 protected:
  Node node_;     ///< Node.
  String frame_;  ///< Frame.
};

}  // namespace hyper::state
