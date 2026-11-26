# usage: bag2npy.py <rosbag dir> <tag>  -> <tag>_markers/_cmd/_fcvel/_rdy.npy in cwd
# cmd columns: t, linear xyz, angular xyz (angular.z = relative yaw step [rad])
import sys, numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
path, tag = sys.argv[1], sys.argv[2]
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
mk, cmd, fv, rdy = [], [], [], []
while r.has_next():
    topic, data, t = r.read_next(); t *= 1e-9
    msg = deserialize_message(data, get_message(types[topic]))
    if topic == '/markers':
        for m in msg.markers:
            mk.append([t, m.id, m.pose.position.x, m.pose.position.y, m.pose.position.z, m.action])
        if not msg.markers: mk.append([t, -1, np.nan, np.nan, np.nan, -1])
    elif topic == '/boids/cmd_vel':
        l, a = msg.twist.linear, msg.twist.angular; cmd.append([t, l.x, l.y, l.z, a.x, a.y, a.z])
    elif topic == '/fc_rdy':
        rdy.append([t, float(msg.data)])
    elif topic == '/fc_vel':
        l, a = msg.twist.linear, msg.twist.angular; fv.append([t, l.x, l.y, l.z, a.x, a.y, a.z])
np.save(tag + '_markers.npy', np.array(mk)); np.save(tag + '_cmd.npy', np.array(cmd)); np.save(tag + '_fcvel.npy', np.array(fv)); np.save(tag + '_rdy.npy', np.array(rdy))
mk = np.array(mk); print('markers rows', len(mk), 'ids', np.unique(mk[:, 1], return_counts=True), 'empty msgs', (mk[:, 1] == -1).sum())
print('span %.1f s' % (max(mk[-1, 0], cmd[-1][0]) - min(mk[0, 0], cmd[0][0])))
