#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/base_scan', self.scan_callback, 10)

        # tuned params
        self.desired_wall_dist = 0.4
        self.forward_speed = 0.2
        self.safe_distance = 0.35
        self.doorway_threshold = 0.6  # More sensitive doorway detection

        self.state = 'find_wall'  # Start by finding a wall
        self.last_regions = None
        
        # doorway state machine
        self.doorway_mode = None
        self.doorway_steps = 0
        
        # corner stuck detection
        self.stuck_counter = 0
        self.last_position_check = 0
        self.was_stuck_recently = False
        self.stuck_recovery_steps = 0
        
        # wall following status
        self.has_wall = False
        self.no_wall_counter = 0

        self.get_logger().info("wall follower started - searching for wall")

    def scan_callback(self, msg: LaserScan):
        data = np.array(msg.ranges)
        data = np.where(np.isinf(data), 10.0, data)
        data = np.where(np.isnan(data), 10.0, data)

        regions = self.get_regions(data)
        self.last_regions = regions
        self.take_action(regions)

    def get_regions(self, data):
        n = len(data)
        
        def safe_min(start_idx, end_idx):
            start_idx = max(0, min(start_idx, n))
            end_idx = max(0, min(end_idx, n))
            if start_idx >= end_idx:
                return 10.0
            return np.min(data[start_idx:end_idx])
        
        return {
            'front': safe_min(n*7//16, n*9//16),
            'front_left': safe_min(n*9//16, n*11//16),
            'left': safe_min(n*11//16, n*13//16),
            'front_right': safe_min(n*5//16, n*7//16),
            'right': safe_min(n*3//16, n*5//16),
            'side_right': safe_min(n*1//16, n*3//16),
            'back_right': safe_min(0, n*1//16),
        }

    def check_if_has_wall(self, r):
        """Check if we currently have a wall to follow on the right"""
        return (r['side_right'] < 1.5 or r['right'] < 1.5 or r['back_right'] < 1.5)
    
    def check_stuck_in_corner(self, r):
        """Detect if stuck in a corner (obstacles on multiple sides)"""
        front_blocked = r['front'] < self.safe_distance
        right_blocked = r['right'] < self.safe_distance
        front_right_blocked = r['front_right'] < self.safe_distance
        
        # Stuck if front and right both blocked
        if front_blocked and (right_blocked or front_right_blocked):
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        return self.stuck_counter > 5

    def take_action(self, r):
        d = self.desired_wall_dist
        
        # STUCK RECOVERY - highest priority
        if self.was_stuck_recently:
            self.stuck_recovery_steps += 1
            if self.stuck_recovery_steps > 20:
                self.was_stuck_recently = False
                self.stuck_recovery_steps = 0
                self.get_logger().info("stuck recovery complete")
            else:
                self.state = 'stuck_recovery'
                self.execute_state()
                return
        
        # CHECK FOR CORNER STUCK
        if self.check_stuck_in_corner(r):
            self.was_stuck_recently = True
            self.stuck_recovery_steps = 0
            self.stuck_counter = 0
            self.get_logger().info("STUCK IN CORNER - initiating recovery")
            self.state = 'stuck_recovery'
            self.execute_state()
            return
        
        # INITIAL WALL FINDING
        if self.state == 'find_wall':
            if self.check_if_has_wall(r):
                self.has_wall = True
                self.state = 'follow_wall'
                self.get_logger().info("wall acquired - starting wall following")
            else:
                self.execute_state()
                return
        
        # Update wall tracking
        self.has_wall = self.check_if_has_wall(r)
        if not self.has_wall:
            self.no_wall_counter += 1
        else:
            self.no_wall_counter = 0
        
        # DOORWAY STATE MACHINE
        if self.doorway_mode is not None:
            self.doorway_steps += 1
            
            if self.doorway_mode == 'entering':
                # Turn into the doorway and go through
                if self.doorway_steps > 12:
                    self.doorway_mode = 'passing'
                    self.doorway_steps = 0
                    self.get_logger().info("passing through doorway")
                self.state = 'doorway_enter'
                self.execute_state()
                return
                
            elif self.doorway_mode == 'passing':
                # Continue straight through
                if self.doorway_steps > 15:
                    self.doorway_mode = 'passed'
                    self.doorway_steps = 0
                    self.get_logger().info("passed doorway - looking for wall")
                self.state = 'doorway_straight'
                self.execute_state()
                return
                
            elif self.doorway_mode == 'passed':
                # check if we found wall again
                if r['side_right'] < d * 1.5 or r['right'] < d * 1.5:
                    self.doorway_mode = None
                    self.doorway_steps = 0
                    self.has_wall = True
                    self.get_logger().info("wall reacquired after doorway")
                elif self.doorway_steps > 25:
                    self.doorway_mode = 'turning'
                    self.doorway_steps = 0
                    self.get_logger().info("actively searching for wall")
                else:
                    self.state = 'doorway_search'
                    self.execute_state()
                    return
                    
            elif self.doorway_mode == 'turning':
                if r['side_right'] < d * 1.5 or r['right'] < d * 1.5:
                    self.doorway_mode = None
                    self.doorway_steps = 0
                    self.has_wall = True
                    self.get_logger().info("wall found after search")
                elif self.doorway_steps > 20:
                    self.doorway_mode = None
                    self.doorway_steps = 0
                    self.get_logger().info("search timeout")
                else:
                    self.state = 'doorway_turn_hard'
                    self.execute_state()
                    return
        
        # DETECT NEW DOORWAY - only when we have a wall and not in doorway mode
        if self.doorway_mode is None and self.has_wall:
            # Doorway = big gap on right, clear front, AND we HAD a wall before
            if (r['side_right'] > self.doorway_threshold and 
                r['right'] > self.doorway_threshold and 
                r['front'] > self.safe_distance * 2.0 and
                self.no_wall_counter < 3):  # Just lost wall
                
                self.doorway_mode = 'entering'
                self.doorway_steps = 0
                self.state = 'doorway_enter'
                self.get_logger().info("DOORWAY DETECTED - turning in")
                self.execute_state()
                return
        
        # OBSTACLE AVOIDANCE - always check front obstacles first!
        if r['front'] < self.safe_distance:
            self.state = 'turn_left'
            
        elif r['front_left'] < self.safe_distance * 0.7:
            # Very close on front-left, stop and turn in place
            self.state = 'rotate_right'
            
        elif r['front_left'] < self.safe_distance:
            self.state = 'avoid_left'
            
        elif r['front_right'] < self.safe_distance:
            self.state = 'avoid_corner'
            
        elif r['left'] < self.safe_distance * 0.8:
            # Too close on left side
            self.state = 'drift_right'
            
        elif not self.has_wall and self.no_wall_counter > 10:
            # Lost wall for a while, search for it
            self.state = 'search_wall'
            
        elif r['side_right'] > d * 2.5 and r['right'] > d * 2.0:
            # Wall drifting away
            self.state = 'turn_right'
            
        elif r['side_right'] < d * 0.5:
            self.state = 'too_close'
            
        else:
            self.state = 'follow_wall'

        self.execute_state()

    def execute_state(self):
        msg = Twist()
        
        if self.state == 'find_wall':
            # Spiral outward to find a wall
            msg.linear.x = self.forward_speed
            msg.angular.z = -0.5
            
        elif self.state == 'stuck_recovery':
            # Back up and turn left to escape corner
            msg.linear.x = -0.05
            msg.angular.z = 0.6
            
        elif self.state == 'doorway_enter':
            # Turn right into doorway
            msg.linear.x = self.forward_speed * 0.6
            msg.angular.z = -0.5
            
        elif self.state == 'doorway_straight':
            msg.linear.x = self.forward_speed
            msg.angular.z = 0.0
            
        elif self.state == 'doorway_search':
            msg.linear.x = self.forward_speed * 0.4
            msg.angular.z = -0.4
            
        elif self.state == 'doorway_turn_hard':
            msg.linear.x = self.forward_speed * 0.5
            msg.angular.z = -0.7
            
        elif self.state == 'turn_left':
            msg.linear.x = 0.05
            msg.angular.z = 0.6
            
        elif self.state == 'rotate_right':
            # Too close on front-left, rotate in place
            msg.linear.x = 0.0
            msg.angular.z = -0.6  # Turn right without moving forward
            
        elif self.state == 'avoid_left':
            # Obstacle on front-left, turn away from it
            msg.linear.x = self.forward_speed * 0.3
            msg.angular.z = -0.5  # Turn right
            
        elif self.state == 'avoid_corner':
            msg.linear.x = self.forward_speed * 0.5
            msg.angular.z = 0.4
            
        elif self.state == 'drift_right':
            # Too close on left, drift right
            msg.linear.x = self.forward_speed * 0.6
            msg.angular.z = -0.3
            
        elif self.state == 'search_wall':
            # Turn right to find wall
            msg.linear.x = self.forward_speed * 0.6
            msg.angular.z = -0.5
            
        elif self.state == 'turn_right':
            msg.linear.x = self.forward_speed * 0.7
            msg.angular.z = -0.5
            
        elif self.state == 'too_close':
            msg.linear.x = self.forward_speed * 0.8
            side_dist = self.last_regions['side_right']
            error = side_dist - self.desired_wall_dist
            msg.angular.z = 1.5 * error
            
        else:  # follow_wall
            msg.linear.x = self.forward_speed
            side_dist = self.last_regions['side_right']
            error = self.desired_wall_dist - side_dist
            k = 1.2
            msg.angular.z = np.clip(k * error, -0.8, 0.8)
        
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()