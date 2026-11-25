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
      twist.angular.z is NOT a rate: it is a relative heading change [rad] (FLU, positive
      = turn left) toward the neighbors' bearing, re-sent with every message; 0 = hold
      the current heading target. The bridge sends it as relative yaw (BODY_FRD, -z).

Rules (all computed in the local frame, so the drone itself sits at 0,0,0):
  cohesion   -> steer toward the neighbors' centroid:        mean(p_i)
                (separate horizontal and vertical gains; the vertical one keeps
                 neighbors inside the camera's narrow vertical view)
  separation -> steer away from neighbors closer than sep_dist:
                sum(-p_i/|p_i| * (1/|p_i| - 1/sep_dist)), horizontal component only
  alignment  -> match neighbors' velocity: v_i - v_self, where v_i is neighbor i's
                ground velocity and v_self is ours (fc_vel), both in body axes (FLU).
                Since p_i is relative, d(p_i)/dt = (v_i - v_self) - w x p_i
                (w = body rate). The filter estimates v_i using the measured v_self
                in its prediction, so the slow camera filter only smooths v_i.
                In the steering, v_self is a blend of the measured velocity (reacts
                to drift, but through our ~0.3 s response) and our own command (no
                delay); damp_measured sets the share. Damping only the measured
                velocity oscillated at 0.4 Hz in the 2026-11-25 flights.
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
        # separation fades to zero at sep_dist; with w_cohesion 0.25 a single neighbor
        # settles at ~2.0 m (2026-11-25 flight delays: camera 0.04 s, fc_vel 0.06 s,
        # our response x 0.46+0.08 s, y 0.12+0.30 s, z 0.30+0.28 s)
        self.sep_dist = p('separation_distance', 2.4).value
        self.w_coh = p('w_cohesion', 0.25).value            # horizontal
        # vertical cohesion (match height); tuned on the 2026-11-19 16:03 flight replay:
        # higher gains chase the apparent height change caused by our own pitch
        self.w_height = p('w_height', 0.75).value
        self.w_sep = p('w_separation', 6.0).value
        self.w_ali = p('w_alignment', 3.0).value              # also damps closing speed
        # share of measured (fc_vel) vs commanded velocity in alignment's v_self.
        # The measured part closes a loop through our ~0.3 s command response
        # (2026-11-25 14:02 log: x 0.10+0.30 s, y 0.08+0.36 s, z 0.28+0.30 s);
        # 1.0 with w_alignment 3 oscillated at 0.4 Hz, 0.5 keeps 40+ deg phase margin
        self.damp_meas = p('damp_measured', 0.5).value
        self.max_speed = p('max_speed', 1.0).value          # [m/s]
        self.max_acc = p('max_accel', 2.0).value            # [m/s^2]
        self.timeout = p('timeout', 1.0).value              # stop if no input [s]
        # yaw tracking: the FC adds each relative heading change onto its own heading
        # target, so the turn rate is (step / message interval). Sending
        # yaw_gain * bearing * dt makes the gain a rate gain [1/s], independent of the
        # marker rate (0 disables)
        self.yaw_gain = p('yaw_gain', 1.5).value
        self.max_yaw_rate = p('max_yaw_rate', 0.5).value    # [rad/s], below ATC_RATE_WPY_MAX
        # alpha-beta filter gains (0..1]: lower = smoother but laggier; 1.0/1.0 = no filtering
        # tuned per update on ~21 Hz WhyCon data (about 1 cm depth noise)
        self.alpha = p('filter_alpha', 0.6).value           # position correction gain
        self.beta = p('filter_beta', 0.2).value             # velocity correction gain
        # anti-windup: only integrate the command while the FC follows it
        self.require_rdy = p('require_fc_ready', True).value

        self.vel = np.zeros(3)   # current commanded velocity (local frame)
        self.tracks = {}         # id -> (filtered position, filtered neighbor velocity, time)
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
        # no (recent) fc_vel -> assume the frame does not rotate and we are not moving
        fc_ok = self.fc_t is not None and now - self.fc_t < self.timeout
        w = self.omega if fc_ok else np.zeros(3)
        va = self.v_meas if fc_ok else np.zeros(3)

        pos, nb_vel, tracks = [], [], {}
        for m in msg.markers:
            if m.action != Marker.ADD:
                continue
            z = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
            q, u, valid_v = z, va.copy(), False   # new track: assume it moves with us
            if m.id in self.tracks:
                q0, u0, t0 = self.tracks[m.id]
                h = now - t0
                if 0.0 < h < self.timeout:
                    # alpha-beta filter on the neighbor's own velocity u (body axes):
                    # predict q with the relative velocity u - va in a frame rotating at w
                    # (q and u rotate by -w x), correct with the residual
                    q_pred = q0 + (u0 - va - np.cross(w, q0)) * h
                    u_pred = u0 - np.cross(w, u0) * h
                    r = z - q_pred
                    q = q_pred + self.alpha * r
                    u = u_pred + (self.beta / h) * r
                    valid_v = True
            tracks[m.id] = (q, u, now)
            if np.linalg.norm(q) > self.radius:
                continue
            pos.append(q)
            if valid_v:
                nb_vel.append(u)
        self.tracks = tracks

        acc = np.zeros(3)
        yaw = 0.0
        if pos:
            P = np.array(pos)
            c = P.mean(axis=0)
            if c[0] > 0.0:
                # turn toward the neighbors' horizontal bearing
                rate = np.clip(self.yaw_gain * np.arctan2(c[1], c[0]),
                               -self.max_yaw_rate, self.max_yaw_rate)
                # cap dt: after a gap one late step would bypass max_yaw_rate
                yaw = float(rate * min(dt, 0.1))
            acc += self.w_coh * np.array([c[0], c[1], 0.0])     # horizontal cohesion
            acc[2] += self.w_height * c[2]                      # height matching
            d = np.linalg.norm(P, axis=1)
            close = (d > 1e-6) & (d < self.sep_dist)
            if close.any():
                # horizontal only: keep apart side by side instead of stacking
                # (a neighbor in the camera view is never far above or below).
                # (1/d - 1/sep_dist) falls to zero at sep_dist: a step there made us
                # bounce in and out of it (no equilibrium with cohesion)
                k = 1.0 / d[close] - 1.0 / self.sep_dist
                sep = (-P[close] / d[close, None] * k[:, None]).sum(axis=0)
                sep[2] = 0.0
                acc += self.w_sep * sep
        if nb_vel:
            # v_self: blend of measured and commanded velocity (see docstring)
            v_self = self.damp_meas * va + (1.0 - self.damp_meas) * self.vel
            acc += self.w_ali * (np.mean(nb_vel, axis=0) - v_self)

        ready = self.fc_ready()
        if not ready:
            # FC is not following (not in Guided): don't let the command wind up
            self.vel = np.zeros(3)
            yaw = 0.0
        else:
            if not self.following:
                # (re)entering Guided: continue from what we are actually doing
                self.vel = clamp(self.v_meas.copy(), self.max_speed) if fc_ok else np.zeros(3)
            # keep the previous command fixed in the world while the body rotates
            self.vel = self.vel - np.cross(w, self.vel) * dt
            self.vel = clamp(self.vel + clamp(acc, self.max_acc) * dt, self.max_speed)
        self.following = ready
        self.publish(self.vel, yaw)

    def watchdog(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_t is not None and now - self.last_t > self.timeout:
            self.vel[:] = 0.0
            self.last_t = None
            self.tracks.clear()
            self.publish(self.vel)

    def publish(self, v, yaw=0.0):
        t = TwistStamped()
        t.header.frame_id = "body"
        t.header.stamp = self.get_clock().now().to_msg()
        t.twist.linear.x, t.twist.linear.y, t.twist.linear.z = map(float, v)
        t.twist.angular.z = float(yaw)
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
