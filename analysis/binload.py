import sys, numpy as np
from pymavlink import mavutil
want = {'PSCN':['DVN','TVN','VN','DAN','TAN','AN','DPN','TPN','PN'],
        'PSCE':['DVE','TVE','VE','DAE','TAE','AE','DPE','TPE','PE'],
        'PSCD':['DVD','TVD','VD','DAD','TAD','AD','DPD','TPD','PD'],
        'XKF1':['PN','PE','PD','VN','VE','VD','Roll','Pitch','Yaw'],
        'GUIP':['vX','vY','vZ'], 'ATT':['Roll','Pitch','Yaw','DesRoll','DesPitch','DesYaw'],
        'GPS':['GMS','GWk','Spd','GCrs','VZ','Status','NSats','HDop','Lat','Lng','Alt'],
        'XKF3':['IVN','IVE','IVD','IPN','IPE','IPD'], 'MODE':['Mode','Rsn'],
        'XKF4':['SV','SP','SH','SS'], 'ORGN':['Lat','Lng','Alt'], 'RATE':['RDes','R','PDes','P','YDes','Y'], 'RCIN':['C1','C2','C3','C4','C8']}
m = mavutil.mavlink_connection(sys.argv[1])  # usage: binload.py <log.bin> <out.npz>
d = {k: [] for k in want}
while True:
    msg = m.recv_match(type=list(want))
    if msg is None: break
    t = msg.get_type()
    if t == 'GPS' and getattr(msg, 'I', 0) != 0: continue
    if t == 'XKF1' and msg.C != 0: continue
    d[t].append([msg.TimeUS * 1e-6] + [getattr(msg, f) for f in want[t]])
np.savez(sys.argv[2], **{k: np.array(v, float) for k, v in d.items()})
for k, v in d.items(): print(k, len(v))
