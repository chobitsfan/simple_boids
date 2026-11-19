#!/usr/bin/env python3
"""Minimal Reynolds (1986) boids controller for a single drone.

Sub:  ~/neighbors (visualization_msgs/MarkerArray)
      one marker per observed neighbor; only `id` and `pose.position` are used,
      position expressed in the drone's local frame (drone at origin).
Sub:  fc_vel (geometry_msgs/TwistStamped), optional; drone's own velocity [m/s] in
      twist.linear and angular rate [rad/s] in twist.angular, same body frame (FLU)
      as the markers.
Sub:  fc_rdy (std_msgs/Bool); true while the flight controller follows velocity
      commands. While it is false (or never received) the command is held at zero
      (anti-windup); when it turns true, the command restarts from fc_vel.
Pub:  ~/cmd_vel (geometry_msgs/TwistStamped), linear velocity in the drone's local frame.

Rules (all computed in the local frame, so the drone itself sits at 0,0,0):
  cohesion   -> steer toward the neighbors' centroid:        mean(p_i)
  separation -> steer away from neighbors that are too close: sum(-p_i / |p_i|^2)
  alignment  -> match neighbors' velocity. Since p_i is relative, d(p_i)/dt is
                (v_i - v_self) - w x p_i, where w is the body rate; the filter
                removes the rotation part, leaving the alignment steering term.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray


class BoidsNode(Node):
    def __init__(self):
        super().__init__('boids')
        p = self.declare_parameter
        self.radius = p('neighbor_radius', 5.0).value     # perception range [m]
        # weights tuned in closed-loop sim with the measured drone response
        # (2026-11-19 DataFlash log: first-order lag 0.82 s + 0.12 s delay in x,
        #  1.0 s + 0.18 s in y; PSC_JERK_NE 5, WP_ACC 2.5)
        self.sep_dist = p('separation_distance', 1.8).value
        self.w_coh = p('w_cohesion', 0.25).value
        self.w_sep = p('w_separation', 4.0).value
        self.w_ali = p('w_alignment', 3.0).value              # also damps closing speed
        self.max_speed = p('max_speed', 1.0).value          # [m/s]
        self.max_acc = p('max_accel', 2.0).value            # [m/s^2]
        self.timeout = p('timeout', 1.0).value              # stop if no input [s]
        # alpha-beta filter gains (0..1]: lower = smoother but laggier; 1.0/1.0 = no filtering
        # tuned per update on ~21 Hz WhyCon data (about 1 cm depth noise)
        self.alpha = p('filter_alpha', 0.6).value           # position correction gain
        self.beta = p('filter_beta', 0.2).value             # velocity correction gain
        # anti-windup: only integrate the command while the FC follows it
        self.require_rdy = p('require_fc_ready', True).value

        self.vel = np.zeros(3)   # current commanded velocity (local frame)
        self.tracks = {}         # id -> (filtered position, filtered relative velocity, time)
        self.last_t = None
        self.omega = np.zeros(3)  # own body rate [rad/s]
        self.v_meas = np.zeros(3)  # own velocity [m/s]
        self.fc_t = None           # time of last fc_vel
        self.rdy = False           # latest fc_rdy
        self.following = False

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(TwistStamped, '~/cmd_vel', qos)
        self.create_subscription(MarkerArray, 'markers', self.on_neighbors, qos)
        self.create_subscription(TwistStamped, 'fc_vel', self.on_fc_vel, qos)
        self.create_subscription(Bool, 'fc_rdy', self.on_fc_rdy, qos)
        self.create_timer(0.1, self.watchdog)

    def on_fc_vel(self, msg: TwistStamped):
        a, v = msg.twist.angular, msg.twist.linear
        self.omega = np.array([a.x, a.y, a.z])
        self.v_meas = np.array([v.x, v.y, v.z])
        self.fc_t = self.get_clock().now().nanoseconds * 1e-9

    def on_fc_rdy(self, msg: Bool):
        self.rdy = msg.data

    def fc_ready(self):
        return self.rdy or not self.require_rdy

    def on_neighbors(self, msg: MarkerArray):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.last_t is None else min(now - self.last_t, self.timeout)
        self.last_t = now
        # no (recent) fc_vel -> assume the frame does not rotate and A is not moving
        fc_ok = self.fc_t is not None and now - self.fc_t < self.timeout
        w = self.omega if fc_ok else np.zeros(3)

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
                    # alpha-beta filter: predict with constant velocity in a frame rotating
                    # at w (both q and v rotate by -w x), correct with residual
                    q_pred = q0 + (v0 - np.cross(w, q0)) * h
                    v_pred = v0 - np.cross(w, v0) * h
                    r = z - q_pred
                    q = q_pred + self.alpha * r
                    v = v_pred + (self.beta / h) * r
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

        ready = self.fc_ready()
        if not ready:
            # FC is not following (not in Guided): don't let the command wind up
            self.vel = np.zeros(3)
        else:
            if not self.following:
                # (re)entering Guided: continue from what A is actually doing
                self.vel = clamp(self.v_meas.copy(), self.max_speed) if fc_ok else np.zeros(3)
            # keep the previous command fixed in the world while the body rotates
            self.vel = self.vel - np.cross(w, self.vel) * dt
            self.vel = clamp(self.vel + clamp(acc, self.max_acc) * dt, self.max_speed)
        self.following = ready
        self.publish(self.vel)

    def watchdog(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_t is not None and now - self.last_t > self.timeout:
            self.vel[:] = 0.0
            self.last_t = None
            self.tracks.clear()
            self.publish(self.vel)

    def publish(self, v):
        t = TwistStamped()
        t.header.frame_id = "body"
        t.header.stamp = self.get_clock().now().to_msg()
        t.twist.linear.x, t.twist.linear.y, t.twist.linear.z = map(float, v)
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
