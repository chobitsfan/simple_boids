"""One-shot flight report for the wingman (and leader, if its log is given).

usage (run in a work dir holding the extracted data, see CLAUDE.md):
  analyze_flight.py <tag> <wingman.npz> [<leader.npz>]
"""
import sys, numpy as np, rclpy
from flight import load_pair, geometry, Rn, F
from replay import replay

tag, wbin = sys.argv[1], sys.argv[2]; lbin = sys.argv[3] if len(sys.argv) > 3 else None
D = load_pair(tag, wbin, lbin); c, c0, X, G, L = D['c'], D['c0'], D['X'], D['G'], D['L']
tc = c[:, 0] - c0; inG = (tc > G) & (tc < L)
print('== clock: cmd vs GUIP rms %.4f (should be < 0.01); Guided %.1f..%.1f s (%.0f s)' % (D['gu_err'], G, L, L - G))

rclpy.init(); o, _ = replay(tag); rclpy.shutdown()
i = np.clip(np.searchsorted(o[:, 0], tc - 1e-6), 0, len(o) - 1); dv = np.linalg.norm(o[i, 1:4] - c[:, 1:4], axis=1)[inG]
print('== replay through repo boids_node.py vs recorded cmd: median %.3f p95 %.3f m/s (~0.05-0.1 = same code; integrator makes it never exact)' % (np.median(dv), np.percentile(dv, 95)))

geo = geometry(D, tag)

# --- oscillation: wingman body velocity spectra
tm, P = D['tm'], D['P']; g = (tm > G + 3) & (tm < L - 1); tm, P = tm[g], P[g]
S = np.array([X(t) for t in tm]); VWb = np.array([(Rn(*s[6:9]).T @ s[3:6]) * F for s in S])
ti = np.arange(tm[0], tm[-1], 0.05); f = np.fft.rfftfreq(len(ti), 0.05)
def band(y):
    yi = np.interp(ti, tm, y); yi = yi - yi.mean(); Sp = abs(np.fft.rfft(yi * np.hanning(len(yi))))**2; m = (f > 0.05) & (f < 3)
    return yi.std(), f[np.argmax(Sp * m)], 100 * Sp[(f > 0.25) & (f < 0.6)].sum() / Sp[m].sum()
print('== wingman oscillation (0.25-0.6 Hz share; ~0.4 Hz = damping loop too fast for our response)')
for k, ax in enumerate('xyz'):
    print('  body v%s rms %.2f m/s, peak %.2f Hz, 0.25-0.6 Hz %.0f%%' % ((ax,) + band(VWb[:, k])))
print('  pitch rms %.1f deg, peak %.2f Hz, 0.25-0.6 Hz %.0f%%' % band(S[:, 7]))

if lbin:
    LP, LV, LY = D['LP'], D['LV'], D['LY']; lt = np.array([LV(t) for t in tm]); ls = np.hypot(lt[:, 0], lt[:, 1])
    q = (ls < 0.15) & (abs(lt[:, 2]) < 0.15); d = np.linalg.norm(P, axis=1)
    print('== leader speed median %.2f p90 %.2f m/s; leader still %.0f%% of time' % (np.median(ls), np.percentile(ls, 90), 100 * q.mean()))
    if q.sum() > 20:
        ws = np.linalg.norm(S[q, 3:5], axis=1)
        print('  while leader still: wingman horiz speed median %.2f p90 %.2f, |vz| median %.2f; distance p5 %.2f p50 %.2f p95 %.2f' % (
            np.median(ws), np.percentile(ws, 90), np.median(abs(S[q, 5])), *np.percentile(d[q], [5, 50, 95])))
    e = np.array([(Rn(*X(t)[6:9]) @ (p * F)) - (LP(t) - X(t)[0:3]) for t, p in zip(tm[::10], P[::10])])
    print('  camera vs EKF relative position (NED) mean %s std %s (horizontal std < ~0.2 = GPS geometry usable)' % (e.mean(0).round(2), e.std(0).round(2)))

# --- our response: cmd (body) -> body velocity, first-order lag + delay; damping-loop margins
tg = np.arange(G + 3, L - 1, 0.02)
VW = np.array([(Rn(*X(t)[6:9]).T @ X(t)[3:6]) * F for t in tg]); U = np.array([np.interp(tg, tc, c[:, k + 1]) for k in range(3)]).T
fits = {}
print('== response cmd -> velocity (body), first-order lag + delay')
for k, ax in enumerate('xyz'):
    best = None
    for tau in np.arange(0.06, 1.2, 0.02):
        for dl in np.arange(0, 0.44, 0.02):
            n = int(round(dl / 0.02)); u = np.r_[np.full(n, U[0, k]), U[:len(U) - n, k]]; v = np.empty_like(u); v[0] = VW[0, k]; a = 0.02 / tau
            for j in range(1, len(u)): v[j] = v[j - 1] + (u[j - 1] - v[j - 1]) * a
            err = np.sqrt(np.mean((v - VW[:, k])**2))
            if best is None or err < best[0]: best = (err, tau, dl)
    fits[ax] = best[1:]; print('  %s: tau %.2f s + delay %.2f s (fit rms %.2f)' % (ax, best[1], best[2], best[0]))
w = np.linspace(0.05, 20, 40000)
def margins(k, tau, dl):
    Lw = k * np.exp(-1j * w * dl) / (1j * w * (1 + 1j * w * tau)); ph = np.degrees(np.unwrap(np.angle(Lw)))
    return 180 + ph[np.argmin(abs(abs(Lw) - 1))], w[np.argmin(abs(ph + 180))] / 2 / np.pi
print('  measured-velocity damping loop (gain = damp_measured * w_alignment, + fc_vel 0.06 s); want PM > ~35 deg:')
for k in (3.0, 2.0, 1.5, 1.0):
    print('   gain %.1f: ' % k + ' | '.join('%s PM %+3.0f deg (%.2f Hz)' % ((ax,) + margins(k, tau, dl + 0.06)) for ax, (tau, dl) in fits.items()))

# --- gaps and their cause
tmg, gaps, big = geo['tm'], geo['gaps'], geo['big']
print('== gaps > 0.3 s (last seen d/az/el/pitch%s)' % (' | EKF geometry in gap, leader heading vs line of sight' if lbin else ''))
for j in big:
    t0, t1 = tmg[j], tmg[j + 1]; line = '  %6.1f-%6.1f (%4.1f s) last d %.2f az %+3.0f el %+3.0f pitch %+5.1f' % (t0, t1, t1 - t0, geo['d'][j], geo['az'][j], geo['el'][j], geo['pitch'][j])
    if lbin:
        ts = np.arange(t0 + 0.05, t1, 0.1); r = []
        for t in ts:
            s = X(t); dn = LP(t) - s[0:3]; rb = (Rn(*s[6:9]).T @ dn) * F
            r.append((np.hypot(rb[0], rb[1]), np.degrees(np.arctan2(rb[1], rb[0])), (LY(t) - np.degrees(np.arctan2(dn[1], dn[0])) + 180) % 360 - 180))
        r = np.array(r)
        line += ' | d %.1f..%.1f az %+.0f..%+.0f heading-vs-LOS %+.0f..%+.0f' % (r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max(), r[:, 2].min(), r[:, 2].max())
    print(line)

if lbin:
    # marker visibility envelope (marker is on the leader's back)
    tmall = D['tm']; rows = []
    for t in np.arange(G + 2, L - 1, 0.1):
        s = X(t); dn = LP(t) - s[0:3]; rb = (Rn(*s[6:9]).T @ dn) * F
        rows.append((np.hypot(rb[0], rb[1]), np.degrees(np.arctan2(rb[1], rb[0])), (LY(t) - np.degrees(np.arctan2(dn[1], dn[0])) + 180) % 360 - 180, np.min(abs(tmall - t)) < 0.1))
    R = np.array(rows); ok = (R[:, 0] < 2.6) & (abs(R[:, 1]) < 20)
    print('== detection rate with leader in front (<20 deg) and < 2.6 m, by leader heading vs line of sight (0 = we see its back)')
    for lo in range(-180, 180, 30):
        m = ok & (R[:, 2] >= lo) & (R[:, 2] < lo + 30)
        if m.sum() >= 5: print('  %+4d..%+4d deg: %3.0f%% (n=%d)' % (lo, lo + 30, 100 * R[m, 3].mean(), m.sum()))

# --- yaw
a = D['b']['ATT']; ta = D['T'](a[:, 0]); m = (ta > G + 1) & (ta < L - 0.5)
yaw = np.unwrap(np.radians(a[m, 3])); des = np.unwrap(np.radians(a[m, 6])); e = np.degrees(((des - yaw) + np.pi) % (2 * np.pi) - np.pi)
print('== yaw: max step %.3f rad (cap 0.05 at default max_yaw_rate); DesYaw-Yaw rms %.1f max %.1f deg; total turned %.0f deg' % (
    abs(c[inG, 6]).max(), np.sqrt(np.mean(e**2)), abs(e).max(), np.degrees(np.sum(abs(np.diff(yaw))))))
