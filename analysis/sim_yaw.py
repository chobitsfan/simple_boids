"""Closed-loop 2D sim with heading: real BoidsNode + ArduCopter yaw chain.
Relative yaw (Guided FIXED): each cmd sets offset; autoyaw target slews at SLEW;
attitude target follows with sqrt shaper (ACC_Y_MAX, SLEW cap, PILOT_Y_RATE_TC 0);
actual heading lags attitude target. Linear vel: world first-order lag.
Markers 21 Hz, image delay LI, cmd latency LC, FOV +-30 deg, range 2.7 m.
Was used to pick yaw_gain 1.5 /s and max_yaw_rate (2026-11-23)."""
import sys, numpy as np, rclpy
from replay import node_class
BoidsNode = node_class()
from visualization_msgs.msg import Marker, MarkerArray

class Clock:
    t = 0.0
    def now(self):
        c = self
        class N:
            nanoseconds = int(c.t * 1e9)
        return N()

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def R(p): c, s = np.cos(p), np.sin(p); return np.array([[c, -s], [s, c]])

def run(b_traj, k, slew_deg=60.0, T=25.0, dt=0.0025, LI=0.08, LC=0.06, tau_v=0.35,
        acc_y=np.radians(306), tau_att=0.05, max_rate=0.8, psi_b0=0.0, jitter=0.0):
    node = BoidsNode()
    node.yaw_gain = k; node.max_yaw_rate = max_rate; node.rdy = True
    rng = np.random.default_rng(0)
    clk = Clock(); node.get_clock = lambda: clk
    out = []  # (t_arrive, vel_body, yaw)
    node.publish = lambda v, yaw=0.0: out.append((clk.t + LC, np.array(v[:2]), yaw))
    slew = np.radians(slew_deg)
    pa, va = np.zeros(2), np.zeros(2)
    psi = psi_t = tau = rt = 0.0; off = 0.0; vcmd_w = np.zeros(2)
    hist = []  # (t, pa, psi, pb)
    next_m, oi = 0.0, 0
    n_out = n = 0; bear, psis = [], []
    for i in range(int(T / dt)):
        t = i * dt; clk.t = t
        pb = b_traj(t)
        hist.append((t, pa.copy(), psi, pb))
        # FC receives commands
        while oi < len(out) and out[oi][0] <= t:
            _, vb, yaw = out[oi]; oi += 1
            vcmd_w = R(psi) @ vb
            off = yaw
        # autoyaw FIXED slew
        step = np.clip(off, -slew * dt, slew * dt); off -= step; tau += step
        # attitude target: sqrt shaper with accel limit
        e = wrap(tau - psi_t)
        rd = np.sign(e) * min(np.sqrt(2 * acc_y * abs(e)), slew)
        rt += np.clip(rd - rt, -acc_y * dt, acc_y * dt); psi_t += rt * dt
        r_act = (psi_t - psi) / tau_att; psi += r_act * dt
        va += (vcmd_w - va) / tau_v * dt; pa += va * dt
        # fc_vel to node (instant)
        node.omega = np.array([0, 0, r_act]); node.v_meas = np.r_[R(-psi) @ va, 0]
        node.fc_t = t
        # camera
        if t >= next_m:
            next_m += 1 / 21.0 * (1 + jitter * rng.uniform(-1, 1))
            j = max(0, i - int(LI / dt)); _, pa_i, psi_i, pb_i = hist[j]
            rel = R(-psi_i) @ (pb_i - pa_i); d = np.linalg.norm(rel)
            ang = np.degrees(np.arctan2(rel[1], rel[0]))
            ma = MarkerArray()
            if d < 2.7 and rel[0] > 0 and abs(ang) < 30:
                m = Marker(); m.id = 1; m.action = Marker.ADD
                m.pose.position.x, m.pose.position.y = float(rel[0]), float(rel[1])
                ma.markers.append(m); node.on_neighbors(ma)
            elif node.last_t is not None and t - node.last_t > node.timeout:
                node.watchdog()
        # truth metrics
        relt = R(-psi) @ (pb - pa); at = np.degrees(np.arctan2(relt[1], relt[0]))
        vis = np.linalg.norm(relt) < 2.7 and relt[0] > 0 and abs(at) < 30
        n += 1; n_out += not vis; bear.append(at); psis.append(np.degrees(psi))
    node.destroy_node()
    return n_out / n, np.array(bear), np.array(psis)

def static(az_deg, d=2.0):
    p = d * np.array([np.cos(np.radians(az_deg)), np.sin(np.radians(az_deg))])
    return lambda t: p

def cross(v=0.5, d=2.0):           # B slides left in A's initial frame
    return lambda t: np.array([d, v * min(t, 8.0)])

def orbit(rad=2.0, w=0.25):        # B circles A's start point (A must rotate to keep it)
    return lambda t: rad * np.array([np.cos(w * t), np.sin(w * t)])

def turn(v=0.5, d=2.0):            # B flies forward then turns 90 deg left over 4 s
    def f(t):
        if t < 3: return np.array([d + v * t, 0.0])
        a = min((t - 3) / 4, 1) * np.pi / 2; r = v * 4 / (np.pi / 2)
        if t < 7: return np.array([d + 3 * v + r * np.sin(a), r * (1 - np.cos(a))])
        return np.array([d + 3 * v + r, r + v * (t - 7)])
    return f

if __name__ == '__main__':
    rclpy.init()
    for LI, jit in ((0.08, 0.0), (0.15, 0.5)):
        for mr in (0.5, 0.8):
            print(f'=== image delay {LI*1000:.0f} ms, jitter {jit}, max_yaw_rate {np.degrees(mr):.0f} deg/s')
            for k in (0.0, 1.0, 1.5, 2.0, 3.0, 4.0):
                _, b, ps = run(static(20), k, max_rate=mr, T=8, LI=LI, jitter=jit)
                tail = b[len(b) // 2:]
                over = -b.min() if b.min() < 0 else 0.0
                t90 = np.argmax(np.abs(b) < 2) * 0.0025
                res = [f'step20: t<2deg {t90:3.1f}s over {over:4.1f} tail {np.sqrt(np.mean(tail**2)):4.1f}']
                for name, tr in [('cross', cross()), ('orbit', orbit()), ('turn', turn())]:
                    fo, b, _ = run(tr, k, max_rate=mr, LI=LI, jitter=jit)
                    res.append(f'{name}: out {fo*100:3.0f}% |b|max {np.abs(b[:int(len(b)*0.9)]).max():3.0f}')
                print(f'k={k:.1f}  ' + '  '.join(res))
    rclpy.shutdown()
