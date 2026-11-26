# boids_test

`boids_node.py` is a single-file ROS 2 (Jazzy, Python) Reynolds boids controller. It flies
quadcopter **A** (the wingman) behind quadcopter **B** (the leader, flown by a human).
A sees B with a forward camera and commands its own body-frame velocity and heading.
The work is iterative: the user flies, sends the logs, we analyze, fix, simulate, commit,
and the user flies again.

## Setup

- **Camera on A:** WhyCon marker detection at ~21 Hz, about 0.04 s delay and ~1 cm depth noise.
  The field of view is about ±30° horizontal and ±20° vertical, and detection works out to ~3 m.
  Detections arrive as `markers` (MarkerArray, B's position in A's body frame, FLU).
- **The marker is on B's back.** It is detected 97–100% of the time when B's heading is within
  ±60° of the line of sight from A, and 0% when A sees B from the side (60–120°). This caused the
  26.7 s loss on 11-25 14:02.
- **Bridge:** `mav_ros2/my_mav_ros2.cpp`. It runs on A's companion computer, is untracked here,
  and is the user's code. It converts `/boids/cmd_vel` into `SET_POSITION_TARGET_LOCAL_NED` with
  `MAV_FRAME_BODY_FRD` and type_mask `0x9C7` (velocity + yaw, yaw rate ignored), sending
  velocity `(x, -y, -z)` and yaw `-angular.z`. It publishes:
  - `fc_vel` (TwistStamped, FLU body velocity and body rates, from ATTITUDE_QUATERNION and
    LOCAL_POSITION_NED, 20 Hz, about 0.06 s latency).
  - `fc_rdy` (Bool, true in Guided, custom_mode 4).
- **ArduPilot:** a 4.7.0 fork in `../ardupilot`, with two user patches.
  - BODY_FRD velocity is rotated with `ahrs.body_to_earth()`.
  - `GCS_MAVLink_Copter.cpp` (~line 1091) also sets `yaw_relative` for `MAV_FRAME_BODY_FRD`.
    **Without this patch the yaw field is read as an absolute heading and A turns toward north.**
- **Relevant FC parameters:** `boids_wingman.param` is a snapshot; the user later set `PSC_JERK_NE` to 10.
  - `GUID_OPTIONS` 4 and `GUID_TIMEOUT` 3.
  - `WP_YAW_BEHAVIOR` 0 (hold yaw).
  - `ATC_RATE_WPY_MAX` 60, which caps Guided yaw at 60°/s; `ATC_ACC_Y_MAX` 306; `PILOT_Y_RATE_TC` 0.
  - `WP_ACC` 2.5.
- **The companion clock is wrong.** Bag folder names don't match the flight date. Align the bag
  to the FC log by cross-correlating |cmd_vel| with GUIP (`analysis/flight.py` does this). The
  leader log is aligned to the wingman log through GPS time.

## Conventions

- Everything in the node uses A's body frame, FLU (x forward, y left, z up).
- `cmd_vel.twist.linear` is the velocity command.
- `cmd_vel.twist.angular.z` is **not a rate.** It is a relative heading change in radians
  (positive = turn left), re-sent with every message, and 0 holds the heading. ArduPilot adds it
  onto its own heading target, and each message *replaces* the unconsumed remainder.
- MAVLink and ArduPilot use FRD/NED, so the bridge negates y, z and yaw.

## Node design, and why

- **Velocity command:** `vel += clamp(acc, max_accel)·dt`, then clamp to `max_speed`. The stored
  command is rotated by `−w×vel·dt` so that it stays fixed in the world while A rotates.
- **Rules,** applied to the filtered neighbor positions `p_i`:
  - **Cohesion:** horizontal `w_cohesion`.
  - **Height matching:** a separate vertical gain, `w_height` 0.75. Higher values chase the
    apparent height change caused by A's own pitch.
  - **Separation:** horizontal only, `w_sep·(1/d − 1/sep_dist)`. It fades to zero at `sep_dist`.
    The earlier step version had no equilibrium with cohesion, so A bounced across `sep_dist`
    (commit `3843777`). With sep 2.4 and weight 6.0, A settles at about 2.0 m.
  - **Alignment:** `w_ali·(u − v_self)`. Its `−v_self` part is the damping that lets the spring
    terms settle.
- **Filter:** an alpha-beta filter (0.6 / 0.2) on position and on the neighbor's *ground* velocity
  `u`, in body axes. The prediction uses `u − v_meas` (from `fc_vel`) and rotates by `−w×`. This
  keeps the filter's ~0.24 s velocity lag out of A's own damping loop, and a bias in `fc_vel`
  cancels out (`3843777`).
- **`damp_measured` 0.5:** `v_self = 0.5·fc_vel + 0.5·command` (`4dae6cf`).
  - Our command→velocity response is about 0.3 s, so damping only the measured velocity at
    gain 3 left 4–14° of phase margin and oscillated at 0.4 Hz.
  - Damping only the command lets gusts push A to about 0.7 m from B in the sim.
- **Yaw:** the step is `clip(yaw_gain·bearing, ±max_yaw_rate) · min(dt, 0.1)`.
  - Scaling by dt makes `yaw_gain` a rate gain in 1/s. A fixed step per message turns at about
    21·k per second and oscillated in the sim.
  - The dt cap stops a single late step after a gap from bypassing `max_yaw_rate`.
  - A relative step was chosen over a yaw-rate command because if the link dies, A stops turning
    instead of spinning until `GUID_TIMEOUT`.
- **Anti-windup:** while `fc_rdy` is false, the command is zero. On entering Guided, it restarts
  from the measured velocity (`require_fc_ready`).
- **Watchdog:** after `timeout` (1 s) with no markers, the command goes to zero and the tracks are
  cleared, so A stops and waits.

### Ideas already rejected

- Reducing height matching when B is close: that causes bottom exits exactly when B is close.
- A fixed yaw step per message.
- Lowering `w_ali` alone: it only halves the oscillation and weakens braking.
- Damping only the commanded velocity: gusts push A too close.

## Measured wingman response (refit every flight; it changes)

| Flight | x | y | z |
|---|---|---|---|
| 11-19 (`PSC_JERK_NE` 5) | 0.82 + 0.12 s | 1.00 + 0.18 s | – |
| 11-19 16:03 (jerk 10) | 0.37 + 0.06 s | 0.32 + 0.12 s | 0.17 + 0.24 s |
| 11-25 09:58 | 0.46 + 0.08 s | 0.12 + 0.30 s | 0.30 + 0.28 s |
| 11-25 14:02 | 0.10 + 0.30 s | 0.08 + 0.36 s | 0.28 + 0.30 s |

Each cell is a first-order time constant plus a delay. ATT.Yaw tracks ATT.DesYaw within about
1–1.6° RMS.

## Flight history

| Flight (FC log time) | Data in repo root | Result → change |
|---|---|---|
| 11-19 10:05 | `rosbag2_2026_11_19-10_05_12`, `2026-11-19 10-18-52.bin` | The bridge typo sent VD as N; response fitted → anti-windup and retune (`ac7ab72`) |
| 11-19 16:03 | `rosbag2_2026_11_19-16_03_59`, `2026-11-19_16-33-38.bin` | 20% out of view, mostly vertical exits → height matching (`b950d93`) |
| 11-20 14:45 | `rosbag2_2026_11_20-14_45_08`, `2026-11-20_15-01-25.bin` | 12% out of view, side exits → yaw tracking (`bb1656a`) |
| 11-23 14:36 | `rosbag2_2026_11_23-14_36_02`, `2026-11-23_15-09-03.bin` | 3% out of view, azimuth RMS 4°: yaw works |
| 11-23 14:40 | `rosbag2_2026_11_23-14_40_32`, `2026-11-23_15-15-45.bin` | 16% out of view: B at 1.1 m/s exceeded `max_speed`, detection dropped out, the watchdog stopped A |
| 11-25 09:58 | `rosbag2_2026_11_23-15_45_39`, `2026-11-25_09-58-12_{wingman,leader}.bin` | 0.4 Hz oscillation with the leader hovering → filter and smooth separation (`3843777`) |
| 11-25 14:02 | `rosbag2_2026_11_25-12_50_15`, `2026-11-25_14-02-02_{wingman,leader}.bin` | Horizontal fixed; vertical 0.4 Hz remained; 26.7 s loss from leader yaw → `damp_measured` (`4dae6cf`, **not yet flown**) |

## Analysing a flight

```bash
source /opt/ros/jazzy/setup.bash      # pymavlink is in user site-packages: don't use python -I
A=/home/chobits/src/boids_test/analysis; W=/tmp/boids_work; mkdir -p $W && cd $W
python3 $A/bag2npy.py /home/chobits/src/boids_test/rosbag2_<...> f1      # -> f1_*.npy
python3 $A/binload.py /home/chobits/src/boids_test/<wingman>.bin w.npz
python3 $A/binload.py /home/chobits/src/boids_test/<leader>.bin l.npz
python3 $A/analyze_flight.py f1 w.npz l.npz      # leader log optional
python3 $A/sim_closed_loop.py --tau 0.10,0.08,0.28 --delay 0.30,0.36,0.30 damp_meas=0.7
```

`analyze_flight.py` prints:
- clock alignment;
- how closely a replay through the repo node matches the flown commands, which tells you
  whether that code was flown;
- out-of-view time and gaps;
- oscillation spectra;
- the response fit and damping-loop phase margins (want more than ~35°);
- the cause of each gap;
- marker visibility against leader heading;
- yaw tracking.

`sim_closed_loop.py` runs the real node against the fitted response and adds gusts. Without gusts,
a marginal loop looks stable. Use `--node old.py` to compare against another version
(`git show <rev>:boids_node.py > old.py`). `replay.py` replays a flight's inputs through any node
version. `sim_yaw.py` simulates the 2D heading loop.

## Open items (most useful first)

1. **Fly `4dae6cf`.** The vertical 0.4 Hz should be gone. If A drifts in wind, raise
   `damp_measured` toward 0.7; if the 0.4 Hz returns, lower it to about 0.3.
2. **Leader marker visibility:** add markers on B's sides, or use a target that can be seen from
   all directions.
3. **Coast through short gaps** (about 1.5–2 s) on the filter's prediction instead of stopping at
   `timeout`. Stopping while B moves away turns short dropouts into long losses.
4. **`max_speed` 1.0 is below the leader's p90 of 1.1 m/s.** Try about 1.5 in the sim first.
5. **Pitch-induced vertical exits** (every remaining one had pitch > 6°): tilt the camera, add a
   gimbal, or compensate for pitch.
6. **`tracks` keeps only the ids seen in the current message,** so a frame that misses an id resets
   its velocity estimate.
7. **Bridge:** add `armed && !landed` to `fc_rdy` (left out on purpose for handheld tests), and fix
   the shared `buf` that is reused for sends inside parsing.

## Working agreements

- Commit only when the user asks. After committing, give a short flight-test checklist.
- Before changing the control law, check it with `sim_closed_loop.py` (with gusts) and with loop
  margins from the latest fitted response.
- Flight data (`*.bin`, `rosbag2_*`) stays local and is git-ignored.
