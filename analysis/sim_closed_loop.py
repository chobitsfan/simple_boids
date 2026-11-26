"""Closed-loop 3D sim (no yaw) of the real BoidsNode against a modeled wingman.

Wingman per body axis: first-order lag + delay (fit them with analyze_flight.py).
Camera 21 Hz, 0.04 s delay, 2 cm noise, FOV +-30 az / +-20 el, 2.7 m; fc_vel 20 Hz,
0.06 s latency. Gusts (Ornstein-Uhlenbeck velocity disturbance) are needed to expose
a barely-stable damping loop; without them a ~5 deg phase margin looks fine.

usage: sim_closed_loop.py [--node FILE] [--tau x,y,z] [--delay x,y,z] [attr=value ...]
  attr = node attribute (w_ali, damp_meas, sep_dist, w_sep, w_coh, w_height, ...).
  Compares the node as given against the same node with the overrides.
"""
import sys, numpy as np, rclpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TwistStamped
from replay import node_class

class Clock:
    t = 0.0
    def now(self):
        c = self
        class N: nanoseconds = int(c.t * 1e9)
        return N()

def run(cls, params, b_traj, tau, delay, T=30.0, dt=0.005, cam_dl=0.04, fv_dl=0.06, noise=0.02, gust=0.0, seed=0):
    node = cls(); node.yaw_gain = 0.0; node.rdy = True; [setattr(node, k, v) for k, v in params.items()]
    clk = Clock(); node.get_clock = lambda: clk
    cmd = np.zeros(3); node.publish = lambda v, yaw=0.0: cmd.__setitem__(slice(None), v)
    rng = np.random.default_rng(seed); tau = np.asarray(tau); nd = (np.asarray(delay) / dt).astype(int)
    pa, va, gv = np.zeros(3), np.zeros(3), np.zeros(3); H = []
    ncam, nfv = int(cam_dl / dt), int(fv_dl / dt); next_m = next_f = 0.0; out = dict(d=[], va=[], t=[])
    for i in range(int(T / dt)):
        t = i * dt; clk.t = t; pb = b_traj(t)
        H.append((pa.copy(), va + gv, pb, cmd.copy()))
        if t >= next_f:
            next_f += 0.05; vv = H[max(0, i - nfv)][1]
            m = TwistStamped(); m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = map(float, vv); node.on_fc_vel(m)
        if t >= next_m:
            next_m += 1 / 21.0; p_, _, b_, _ = H[max(0, i - ncam)]; rel = b_ - p_ + rng.normal(0, noise, 3)
            d = np.linalg.norm(rel); ma = MarkerArray()
            if d < 2.7 and rel[0] > 0 and abs(np.degrees(np.arctan2(rel[1], rel[0]))) < 30 and abs(np.degrees(np.arcsin(rel[2] / d))) < 20:
                mm = Marker(); mm.id = 1; mm.action = Marker.ADD
                mm.pose.position.x, mm.pose.position.y, mm.pose.position.z = map(float, rel); ma.markers.append(mm); node.on_neighbors(ma)
            elif node.last_t is not None and t - node.last_t > node.timeout:
                node.watchdog()
        u = np.array([H[max(0, i - nd[k])][3][k] for k in range(3)])
        if gust: gv = gv + (-gv / 2.0 + gust * rng.normal(0, 1, 3) / np.sqrt(dt)) * dt
        va += (u - va) / tau * dt; pa += (va + gv) * dt
        out['d'].append(np.linalg.norm(pb - pa)); out['va'].append(va + gv); out['t'].append(t)
    node.destroy_node()
    return {k: np.array(v) for k, v in out.items()}

static = lambda t: np.array([2.0, 0.2, 0.1])
approach = lambda v: (lambda t: np.array([2.2 - v * min(t, 5.0), 0.0, 0.0]))
depart = lambda v: (lambda t: np.array([2.0 + v * min(t, 4.0), 0.0, 0.0]))
weave = lambda t: np.array([2.0, 0.6 * np.sin(0.6 * t), 0.3 * np.sin(0.42 * t)])

def report(name, cls, p, tau, delay):
    g = run(cls, p, static, tau, delay, gust=0.15, T=60); m = g['t'] > 8; v = g['va'][m]; f = np.fft.rfftfreq(len(v), 0.005); osc = []
    for k in range(3):
        S = abs(np.fft.rfft((v[:, k] - v[:, k].mean()) * np.hanning(len(v))))**2; osc.append(np.sqrt(S[(f > 0.3) & (f < 0.7)].sum() / S[f > 0.02].sum()) * v[:, k].std())
    app = min(run(cls, p, approach(s), tau, delay, T=12)['d'].min() for s in (0.3, 0.5, 0.8))
    dep = [run(cls, p, depart(s), tau, delay, T=12)['d'].max() for s in (0.5, 0.8)]
    w = run(cls, p, weave, tau, delay); wm = w['t'] > 5
    print('%-30s gusts+still: 0.3-0.7 Hz vel rms x %.2f y %.2f z %.2f, dist %.2f..%.2f | closest %.2f | depart max %.2f/%.2f | weave %.2f..%.2f' % (
        name, *osc, g['d'][m].min(), g['d'][m].max(), app, *dep, w['d'][wm].min(), w['d'][wm].max()))

if __name__ == '__main__':
    args = sys.argv[1:]; node_file = None; tau, delay = (0.10, 0.08, 0.28), (0.30, 0.36, 0.30)   # 2026-11-25 14:02 fit
    over = {}
    while args:
        a = args.pop(0)
        if a == '--node': node_file = args.pop(0)
        elif a == '--tau': tau = tuple(map(float, args.pop(0).split(',')))
        elif a == '--delay': delay = tuple(map(float, args.pop(0).split(',')))
        else: k, v = a.split('='); over[k] = float(v)
    rclpy.init(); cls = node_class(node_file)
    print('response tau %s delay %s' % (tau, delay))
    report('node as is', cls, {}, tau, delay)
    if over: report(' '.join('%s=%g' % kv for kv in over.items()), cls, over, tau, delay)
    rclpy.shutdown()
