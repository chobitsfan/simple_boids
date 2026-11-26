"""Open-loop replay of recorded inputs (markers, fc_vel, fc_rdy) through a BoidsNode.

replay(tag, params, node_file) -> (out, terms)
  out:   rows (t, cmd x, y, z, yaw step) as published by the node
  terms: rows (t, cohesion xyz, height xyz, separation xyz, alignment xyz) [m/s^2]
node_file defaults to the repo's boids_node.py; pass another file (e.g. `git show
<rev>:boids_node.py > old.py`) to replay an older version. Params are node attribute
names (w_ali, sep_dist, damp_meas, ...), not ROS parameter names.
"""
import os, sys, importlib.util, numpy as np, rclpy
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def node_class(node_file=None):
    f = node_file or os.path.join(REPO, 'boids_node.py')
    spec = importlib.util.spec_from_file_location('boids_node_' + str(abs(hash(f))), f)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod.BoidsNode
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
class Clock:
    t = 0.0
    def now(self):
        k = self
        class N: nanoseconds = int(k.t * 1e9)
        return N()
def replay(tag, params={}, node_file=None):
    mk = np.load(tag + '_markers.npy'); fv = np.load(tag + '_fcvel.npy'); rdy = np.load(tag + '_rdy.npy'); c0 = np.load(tag + '_cmd.npy')[0, 0]
    n = node_class(node_file)(); [setattr(n, k, v) for k, v in params.items()]
    clk = Clock(); n.get_clock = lambda: clk; out, terms = [], []
    n.publish = lambda v, yaw=0.0: out.append((clk.t, *v, yaw))
    ev = [(t - c0, 'm', t) for t in np.unique(mk[:, 0])] + [(r[0] - c0, 'f', r) for r in fv] + [(r[0] - c0, 'r', r[1]) for r in rdy]
    ev.sort(key=lambda e: e[0]); wd = -1
    for t, k, d in ev:
        while wd <= t: clk.t = wd; n.watchdog(); wd += 0.1
        clk.t = t
        if k == 'm':
            ma = MarkerArray()
            for r in mk[mk[:, 0] == d]:
                m = Marker(); m.id = int(r[1]); m.action = int(r[5]); m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, r[2:5]); ma.markers.append(m)
            n.on_neighbors(ma)
            # decompose rule terms from the updated tracks
            Q = [q for q, v, _ in n.tracks.values() if np.linalg.norm(q) <= n.radius]
            tr = np.zeros((4, 3))
            if Q:
                Pq = np.array(Q); cc = Pq.mean(0); tr[0] = n.w_coh * np.r_[cc[0], cc[1], 0]; tr[1, 2] = n.w_height * cc[2]
                dd = np.linalg.norm(Pq, axis=1); cl = (dd > 1e-6) & (dd < n.sep_dist)
                if cl.any(): s = (-Pq[cl] / dd[cl, None] ** 2).sum(0); s[2] = 0; tr[2] = n.w_sep * s
            tr[3] = acc_ali(n)
            terms.append((t, *tr.ravel()))
        elif k == 'f':
            x = TwistStamped(); x.twist.linear.x, x.twist.linear.y, x.twist.linear.z = map(float, d[1:4]); x.twist.angular.x, x.twist.angular.y, x.twist.angular.z = map(float, d[4:7]); n.on_fc_vel(x)
        else: n.on_fc_rdy(Bool(data=bool(d)))
    n.destroy_node()
    return np.array(out), np.array(terms)
def acc_ali(n):
    """alignment term as the current node computes it (u = neighbor velocity estimate)"""
    U = [u for q, u, _ in n.tracks.values() if np.linalg.norm(q) <= n.radius]
    if not U: return np.zeros(3)
    va = n.v_meas; m = getattr(n, 'damp_meas', 1.0)
    return n.w_ali * (np.mean(U, 0) - (m * va + (1 - m) * n.vel))
