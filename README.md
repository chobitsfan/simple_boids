# boids_node

A minimal Reynolds boids controller for a single drone. It runs as a ROS 2 (Jazzy) Python node
on the wingman's companion computer. Each time the camera reports the neighbors' positions,
the node computes cohesion, separation, alignment and height matching. It turns those into a
body-frame velocity command, plus a heading change that keeps the neighbors centred in the
camera view.

Everything is in the drone's own body frame, FLU (x forward, y left, z up), with the drone at
the origin.

## Interface

| | Topic | Type | Content |
|---|---|---|---|
| Sub | `markers` | `visualization_msgs/MarkerArray` | One marker per detected neighbor. Only `id`, `action` and `pose.position` are used. Position is in the body frame. |
| Sub | `fc_vel` | `geometry_msgs/TwistStamped` | Our velocity (`linear`, m/s) and body rate (`angular`, rad/s), FLU. Optional, but needed for good damping and rotation compensation. |
| Sub | `fc_rdy` | `std_msgs/Bool` | True while the flight controller follows commands (Guided). |
| Pub | `~/cmd_vel` → `/boids/cmd_vel` | `geometry_msgs/TwistStamped` | `linear`: velocity command, FLU. `angular.z`: **relative heading change in radians, not a rate** (positive = turn left; 0 = hold heading). |

The bridge (`mav_ros2/`) forwards `cmd_vel` to ArduPilot as `SET_POSITION_TARGET_LOCAL_NED` in
`MAV_FRAME_BODY_FRD`, using relative yaw. **ArduPilot must be patched to treat yaw in BODY_FRD as
relative.** Stock firmware reads it as an absolute heading, and the drone turns toward north.
See [CLAUDE.md](CLAUDE.md).

## Running

```bash
source /opt/ros/jazzy/setup.bash
python3 boids_node.py --ros-args -r markers:=/your/markers -p w_alignment:=3.0 -p max_speed:=1.0
```

Parameters are read **once at startup**. Changing them with `ros2 param set` has no effect, so
restart the node instead.

## How the command is built

Each time markers arrive:

1. **Filter.** Each neighbor's position is tracked with an alpha-beta filter, which also estimates
   that neighbor's own ground velocity `u`. The prediction accounts for our velocity and rotation
   from `fc_vel`.
2. **Acceleration.** The rules give an acceleration:
   ```
   acc = w_cohesion · centroid_xy                                 (pull toward neighbors)
       + w_height   · centroid_z                                  (match height)
       + w_separation · Σ −p̂·(1/d − 1/separation_distance)        (push apart; horizontal only, d < separation_distance)
       + w_alignment  · (mean u − v_self)                         (match velocity; also the damping)
   v_self = damp_measured · fc_vel + (1 − damp_measured) · current command
   ```
3. **Velocity command.** `cmd += clamp(acc, max_accel)·dt`, then clamp to `max_speed`. The stored
   command is rotated as we turn, so it stays fixed in the world.
4. **Heading step.** `clamp(yaw_gain · bearing, ±max_yaw_rate) · min(dt, 0.1)`. The bearing is
   the neighbors' centroid direction.

## Parameters

### Following behaviour

| Parameter | Default | Unit | What it does |
|---|---|---|---|
| `neighbor_radius` | 5.0 | m | Ignore detections farther than this. |
| `w_cohesion` | 0.25 | 1/s² | Horizontal pull toward the neighbors' centroid, per metre of offset. |
| `w_height` | 0.75 | 1/s² | Vertical pull: matches the neighbor's height so it stays inside the narrow (±20°) vertical view. Higher values chase the apparent height change caused by our own pitch. |
| `separation_distance` | 2.4 | m | Range where separation acts. It fades to zero at this distance. |
| `w_separation` | 6.0 | m²/s² | Strength of the horizontal push away. It grows as 1/d at close range. |
| `w_alignment` | 3.0 | 1/s | How fast our velocity converges to the neighbors' velocity. It is also the braking that makes the pull and push settle, and it slows us when a neighbor approaches. |
| `damp_measured` | 0.5 | 0–1 | Share of *measured* velocity (`fc_vel`) vs our own *command* in `v_self`. Measured velocity corrects drift (wind), but it reaches us through the drone's ~0.3 s response. With too much of it, the braking overshoots and the drone oscillates at ~0.4 Hz. The command has no delay, but it ignores drift. |

**Following distance.** Cohesion and separation balance at one distance d*, where
`w_cohesion·d = w_separation·(1/d − 1/separation_distance)`. With the defaults, d* ≈ 2.0 m.

| Change | New d* |
|---|---|
| `separation_distance` 2.2 / `w_separation` 8.0 | ≈ 1.94 m |
| `separation_distance` 2.6 / `w_separation` 5.0 | ≈ 2.05 m |
| `w_cohesion` 0.5 (defaults otherwise) | ≈ 1.77 m |

**Smoothness.** The spring stiffness around d* is `k = w_cohesion + w_separation/d*²` (1.75 /s²
with the defaults). The damping ratio is `w_alignment / (2·√k)`, which is about 1.1 with the
defaults: no overshoot. Below about 0.7 the drone will swing around d*.

**Stability limit.** `damp_measured × w_alignment` is the gain of the loop that passes through
the drone's response delay. With the ~0.3 s delay measured on 2026-11-25:

| `damp_measured × w_alignment` | Phase margin |
|---|---|
| 3.0 | 4–14° (oscillated) |
| 1.5 | 40–51° (default) |
| 1.0 | 55–64° |

Keep the product at about 1.5 or less unless the drone's response gets faster.

### Speed limits

| Parameter | Default | Unit | What it does |
|---|---|---|---|
| `max_speed` | 1.0 | m/s | Limit on the velocity command. If the leader flies faster than this, we fall behind and lose it at range. |
| `max_accel` | 2.0 | m/s² | Limit on how fast the command changes. |

### Heading (yaw tracking)

| Parameter | Default | Unit | What it does |
|---|---|---|---|
| `yaw_gain` | 1.5 | 1/s | Turn rate per radian of bearing: a neighbor 10° off-centre gives about 15°/s. Lower it (e.g. 1.0) if the heading swings back and forth around the neighbor. 0 disables yaw tracking (it sends 0 = hold). |
| `max_yaw_rate` | 0.5 | rad/s | Turn-rate cap (29°/s). Keep it below ArduPilot's `ATC_RATE_WPY_MAX` (60°/s). The step per message is also capped at `max_yaw_rate × 0.1 s`. |

### Filter

| Parameter | Default | Unit | What it does |
|---|---|---|---|
| `filter_alpha` | 0.6 | 0–1 | How much of each new measurement goes into the position estimate. Lower is smoother but lags more. |
| `filter_beta` | 0.2 | 0–1 | How much each new measurement corrects the neighbor-velocity estimate. Lower is smoother but slower to notice when the neighbor changes speed. Since 2026-11-25 this lag affects only the leader's velocity estimate, not our own damping, so lowering it is safe for stability. |

Both are tuned for ~21 Hz WhyCon detections with about 1 cm depth noise. 1.0 / 1.0 means no
filtering.

### Safety

| Parameter | Default | Unit | What it does |
|---|---|---|---|
| `timeout` | 1.0 | s | If no markers arrive for this long, the command goes to zero (hover) and the tracks are cleared. It is also the maximum age of `fc_vel` that is still used. |
| `require_fc_ready` | true | bool | Hold the command at zero while `fc_rdy` is false. When Guided starts, the command continues from the measured velocity instead of a stale one (anti-windup). Set it to false only for bench tests without the bridge. |

## Known limitations

- **Narrow detection angle on the leader.** The marker is on the leader's back, so it is detected
  only when the leader's heading is within about ±60° of the line from us. When the leader turns
  away, detection stops, and after `timeout` we stop and wait.
- **Pitch moves the neighbor in the image.** Nose-down pitch moves it up in the image, and strong
  pitch (> 6°) can push it out of the vertical view.
- **Track resets.** A track keeps only the ids seen in the current message, so a frame that misses
  an id resets that neighbor's velocity estimate.

## Analysis and simulation

[analysis/](analysis/) contains:
- a one-command flight report: `analyze_flight.py`;
- a replay of recorded inputs through any node version: `replay.py`;
- a closed-loop sim with the measured drone response and gusts: `sim_closed_loop.py`, for trying
  parameter changes before flying.

Usage is in [CLAUDE.md](CLAUDE.md).
