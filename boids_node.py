#!/usr/bin/env python3
"""Minimal Reynolds (1986) boids controller for a single drone.

Sub:  ~/neighbors (visualization_msgs/MarkerArray)
      one marker per observed neighbor; only `id` and `pose.position` are used,
      position expressed in the drone's local frame (drone at origin).
Pub:  ~/cmd_vel (geometry_msgs/Twist), linear velocity in the drone's local frame.

Rules (all computed in the local frame, so the drone itself sits at 0,0,0):
  cohesion   -> steer toward the neighbors' centroid:        mean(p_i)
  separation -> steer away from neighbors that are too close: sum(-p_i / |p_i|^2)
  alignment  -> match neighbors' velocity. Since p_i is relative, d(p_i)/dt is
                (v_i - v_self), which is exactly the alignment steering term.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray


class BoidsNode(Node):
    def __init__(self):
        super().__init__('boids')
        p = self.declare_parameter
        self.radius = p('neighbor_radius', 10.0).value     # perception range [m]
        self.sep_dist = p('separation_distance', 2.0).value
        self.w_coh = p('w_cohesion', 0.5).value
        self.w_sep = p('w_separation', 2.0).value
        self.w_ali = p('w_alignment', 1.0).value
        self.max_speed = p('max_speed', 2.0).value          # [m/s]
        self.max_acc = p('max_accel', 2.0).value            # [m/s^2]
        self.timeout = p('timeout', 0.5).value              # stop if no input [s]
        # alpha-beta filter gains (0..1]: lower = smoother but laggier; 1.0/1.0 = no filtering
        self.alpha = p('filter_alpha', 0.5).value           # position correction gain
        self.beta = p('filter_beta', 0.1).value             # velocity correction gain

        self.vel = np.zeros(3)   # current commanded velocity (local frame)
        self.tracks = {}         # id -> (filtered position, filtered relative velocity, time)
        self.last_t = None

        self.pub = self.create_publisher(Twist, '~/cmd_vel', 10)
        self.create_subscription(MarkerArray, '~/neighbors', self.on_neighbors, 10)
        self.create_timer(0.1, self.watchdog)

    def on_neighbors(self, msg: MarkerArray):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.last_t is None else min(now - self.last_t, self.timeout)
        self.last_t = now

        pos, rel_vel, tracks = [], [], {}
        for m in msg.markers:
            if m.action != Marker.ADD:
                continue
            z = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
            q, v, valid_v = z, np.zeros(3), False
            if m.id in self.tracks:
                q0, v0, t0 = self.tracks[m.id]
                h = now - t0
                if 0.0 < h < self.timeout:
                    # alpha-beta filter: predict with constant velocity, correct with residual
                    q_pred = q0 + v0 * h
                    r = z - q_pred
                    q = q_pred + self.alpha * r
                    v = v0 + (self.beta / h) * r
                    valid_v = True
            tracks[m.id] = (q, v, now)
            if np.linalg.norm(q) > self.radius:
                continue
            pos.append(q)
            if valid_v:
                rel_vel.append(v)
        self.tracks = tracks

        acc = np.zeros(3)
        if pos:
            P = np.array(pos)
            acc += self.w_coh * P.mean(axis=0)
            d = np.linalg.norm(P, axis=1)
            close = (d > 1e-6) & (d < self.sep_dist)
            if close.any():
                acc += self.w_sep * (-P[close] / d[close, None] ** 2).sum(axis=0)
        if rel_vel:
            acc += self.w_ali * np.mean(rel_vel, axis=0)

        self.vel = clamp(self.vel + clamp(acc, self.max_acc) * dt, self.max_speed)
        self.publish(self.vel)

    def watchdog(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_t is not None and now - self.last_t > self.timeout:
            self.vel[:] = 0.0
            self.last_t = None
            self.tracks.clear()
            self.publish(self.vel)

    def publish(self, v):
        t = Twist()
        t.linear.x, t.linear.y, t.linear.z = map(float, v)
        self.pub.publish(t)


def clamp(v, limit):
    n = np.linalg.norm(v)
    return v * (limit / n) if n > limit else v


def main():
    rclpy.init()
    rclpy.spin(BoidsNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
