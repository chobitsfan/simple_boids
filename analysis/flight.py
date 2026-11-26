"""Load a wingman flight (bag npy + DataFlash npz) on one clock, plus optional leader log.

Bag time and FC time are aligned by cross-correlating |cmd_vel| with the FC's GUIP
(the companion clock is wrong). All returned times are seconds since the first cmd_vel.
Leader log is aligned to the wingman log through GPS time (GPS.GMS).
"""
import numpy as np
def Rn(r, p_, y):
    r, p_, y = np.radians([r, p_, y]); cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p_), np.sin(p_), np.cos(y), np.sin(y)
    return np.array([[cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy], [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy], [-sp, sr*cp, cr*cp]])
F = np.array([1, -1, -1])
def load(tag, binf):
    b = dict(np.load(binf)); c = np.load(tag + '_cmd.npy'); c0 = c[0, 0]
    gu = b['GUIP']; modes = b['MODE']; G = modes[modes[:, 1] == 4][0, 0]; after = modes[modes[:, 0] > G]
    L = after[0, 0] if len(after) else b['XKF1'][-1, 0]   # end of (first) Guided
    sc = np.linalg.norm(c[:, 1:4], axis=1); sg = np.linalg.norm(gu[:, 1:4], axis=1)
    def err(off):
        tt = gu[:, 0] + off; m = (tt > c[0, 0]) & (tt < c[-1, 0])
        return np.sqrt(np.mean((np.interp(tt[m], c[:, 0], sc) - sg[m])**2)) if m.sum() > 0.5 * len(gu) else 9
    offs = np.arange(c0 - gu[0, 0] - 300, c0 - gu[0, 0] + 300, 0.02)
    off = offs[np.argmin([err(o) for o in offs])]
    fine = np.arange(off - 0.05, off + 0.05, 0.001); off = fine[np.argmin([err(o) for o in fine])]
    T = lambda tb: tb + off - c0
    x = b['XKF1']; tx = T(x[:, 0]); x[:, 9] = np.degrees(np.unwrap(np.radians(x[:, 9])))
    X = lambda t: np.array([np.interp(t, tx, x[:, k]) for k in range(1, 10)])
    mk = np.load(tag + '_markers.npy'); id1 = mk[:, 1] == 1; tm = mk[id1, 0] - c0; P = mk[id1, 2:5]
    return dict(b=b, c=c, c0=c0, T=T, G=T(G), L=T(L), X=X, tm=tm, P=P, off=off, gu_err=err(off), mk=mk)
def geometry(D, label):
    tm, P, X, G, L = D['tm'], D['P'], D['X'], D['G'], D['L']
    g = (tm > G + 1) & (tm < L - 0.5); tm, P = tm[g], P[g]
    S = np.array([X(t) for t in tm])
    rel_ned = np.array([Rn(*s[6:9]) @ (p * F) for s, p in zip(S, P)])
    d = np.linalg.norm(P, axis=1); el = np.degrees(np.arcsin(P[:, 2] / d)); az = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    dz = -rel_ned[:, 2]; pitch = S[:, 7]
    dur = L - 0.5 - (G + 1); gaps = np.diff(tm); big = np.flatnonzero(gaps > 0.3)
    pb = S[:, 0:3] + rel_ned
    vbh, vbz = [], []
    for i in range(len(tm) - 6):
        if tm[i + 6] - tm[i] < 0.6:
            v = (pb[i + 6] - pb[i]) / (tm[i + 6] - tm[i]); vbh.append(np.hypot(v[0], v[1])); vbz.append(abs(v[2]))
    mins = [i for i in range(5, len(d) - 5) if d[i] == d[i-5:i+6].min() and d[i] < 1.8]
    print('== %s: Guided %.0f s' % (label, dur))
    print('  out of view: %.0f%% of time, %d gaps >0.3 s, longest %.1f s, gaps >1.0 s: %d' % (100 * gaps[big].sum() / dur, len(big), gaps.max(), (gaps > 1.0).sum()))
    at = lambda i: (el[i], pitch[i], az[i], dz[i])
    lossv = np.array([at(i) for i in big]) if len(big) else np.zeros((0, 4))
    print('  at loss: vertical exits (|el|>15) %d, of which A pitch >6 deg %d; side exits (|az|>25) %d; range (d>2.5) %d' % (
        (abs(lossv[:, 0]) > 15).sum(), ((abs(lossv[:, 0]) > 15) & (abs(lossv[:, 1]) > 6)).sum(), (abs(lossv[:, 2]) > 25).sum(), sum(d[i] > 2.5 for i in big)))
    print('  height of B above A: p5 %+.2f median %+.2f p95 %+.2f m, |dz| rms %.2f' % (*np.percentile(dz, [5, 50, 95]), np.sqrt(np.mean(dz**2))))
    print('  camera elevation of B: p5 %+.0f median %+.0f p95 %+.0f deg; A pitch p5 %+.1f p95 %+.1f' % (*np.percentile(el, [5, 50, 95]), *np.percentile(pitch, [5, 95])))
    print('  distance: min %.2f p5 %.2f median %.2f p95 %.2f; time <1.5 m %.1f s; approaches <1.8 m: %d' % (d.min(), *np.percentile(d, [5, 50, 95]), (np.diff(tm, append=tm[-1]) * (d < 1.5)).sum(), len(mins)))
    print('  B speed: horizontal median %.2f p90 %.2f max %.2f; vertical median %.2f p90 %.2f max %.2f m/s' % (np.median(vbh), np.percentile(vbh, 90), max(vbh), np.median(vbz), np.percentile(vbz, 90), max(vbz)))
    return dict(tm=tm, d=d, el=el, az=az, dz=dz, pitch=pitch, big=big, gaps=gaps, pbz=pb[:, 2], paz=S[:, 2])

R_E = 6378137.0
def load_pair(tag, wbin, lbin=None):
    """load() plus leader functions LP (NED pos in wingman EKF frame), LV (NED vel), LY (yaw deg)."""
    D = load(tag, wbin)
    if lbin is None:
        return D
    bw, bl = D['b'], dict(np.load(lbin)); T = D['T']
    gw, gl = bw['GPS'], bl['GPS']
    fw = np.polyfit(gw[:, 1], gw[:, 0], 1); fl = np.polyfit(gl[:, 0], gl[:, 1], 1)
    TL = lambda tl: T(np.polyval(fw, np.polyval(fl, tl)))
    xl = bl['XKF1']; tl = TL(xl[:, 0]); ow, ol = bw['ORGN'][0, 1:4], bl['ORGN'][0, 1:4]
    dn = np.radians(ol[0] - ow[0]) * R_E; de = np.radians(ol[1] - ow[1]) * R_E * np.cos(np.radians(ow[0]))
    dd = -(ol[2] - ow[2])   # vertical origin offset is unreliable (baro); use camera for height
    yl = np.degrees(np.unwrap(np.radians(xl[:, 9])))
    D['LP'] = lambda t: np.array([np.interp(t, tl, xl[:, 1]) + dn, np.interp(t, tl, xl[:, 2]) + de, np.interp(t, tl, xl[:, 3]) + dd])
    D['LV'] = lambda t: np.array([np.interp(t, tl, xl[:, k]) for k in (4, 5, 6)])
    D['LY'] = lambda t: np.interp(t, tl, yl)
    return D
