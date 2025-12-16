# Route Dominance Scoring Formulas

This document explains the mathematical formulas used to calculate the key metrics in the route dominance scoring system.

## 1. Time to Ball (Receiver)

**Formula:**
```
time_to_ball = distance_to_ball / speed_toward_ball
```

**Step-by-step calculation:**

1. **Distance to ball:**
   ```
   to_ball_x = ball_land_x - receiver_x
   to_ball_y = ball_land_y - receiver_y
   to_ball_dist = √(to_ball_x² + to_ball_y²)
   ```

2. **Unit vector toward ball:**
   ```
   to_ball_unit_x = to_ball_x / to_ball_dist
   to_ball_unit_y = to_ball_y / to_ball_dist
   ```

3. **Speed component toward ball (dot product of velocity and unit vector):**
   ```
   speed_toward_ball = receiver_vx × to_ball_unit_x + receiver_vy × to_ball_unit_y
   ```

4. **Time to ball:**
   ```
   if speed_toward_ball > 0:
       time_to_ball = to_ball_dist / speed_toward_ball
   else:
       time_to_ball = ∞  (receiver moving away from ball)
   ```

**Special cases:**
- If `to_ball_dist = 0`: `time_to_ball = 0` (receiver is at ball location)
- If `speed_toward_ball ≤ 0`: `time_to_ball = ∞` (receiver not moving toward ball)

---

## 2. Defender Time to Ball

**Formula:**
```
def_time_to_ball = def_distance_to_ball / def_speed_toward_ball
```

**Step-by-step calculation:**

1. **Distance from defender to ball:**
   ```
   def_to_ball_x = ball_land_x - defender_x
   def_to_ball_y = ball_land_y - defender_y
   def_to_ball_dist = √(def_to_ball_x² + def_to_ball_y²)
   ```

2. **Unit vector from defender to ball:**
   ```
   def_to_ball_unit_x = def_to_ball_x / def_to_ball_dist
   def_to_ball_unit_y = def_to_ball_y / def_to_ball_dist
   ```

3. **Defender velocity components:**
   - If `vx` and `vy` are available:
     ```
     def_vx = defender["vx"]
     def_vy = defender["vy"]
     ```
   - If not available, calculate from speed and direction:
     ```
     def_vx = defender_speed × cos(defender_direction_radians)
     def_vy = defender_speed × sin(defender_direction_radians)
     ```

4. **Defender speed component toward ball:**
   ```
   def_speed_toward_ball = def_vx × def_to_ball_unit_x + def_vy × def_to_ball_unit_y
   ```

5. **Defender time to ball:**
   ```
   if def_speed_toward_ball > 0:
       def_time_to_ball = def_to_ball_dist / def_speed_toward_ball
   else:
       def_time_to_ball = ∞  (defender moving away from ball)
   ```

**Special cases:**
- If `def_to_ball_dist = 0`: `def_time_to_ball = 0`
- If `def_speed_toward_ball ≤ 0`: `def_time_to_ball = ∞`

---

## 3. Time Advantage

**Formula:**
```
time_advantage = def_time_to_ball - time_to_ball
```

**Interpretation:**
- **Positive value**: Receiver reaches ball before defender (advantage for receiver)
- **Negative value**: Defender reaches ball before receiver (advantage for defender)
- **Zero**: Both reach ball at the same time

**Example:**
- If `time_to_ball = 2.5 seconds` and `def_time_to_ball = 3.2 seconds`
- Then `time_advantage = 3.2 - 2.5 = 0.7 seconds` (receiver has 0.7 second advantage)

**Special cases:**
- If either time is `∞`, then `time_advantage = ∞`

---

## 4. Dominance Score

**Formula:**
```
dominance_score = 0.25 × sep_score + 
                 0.18 × speed_score + 
                 0.12 × accel_score + 
                 0.18 × time_score + 
                 0.12 × pressure_score + 
                 0.15 × leverage_score
```

**Component Calculations:**

### 4.1 Separation Score
```
sep_score = min(sep_nearest / 10.0, 1.0)
```
- Normalizes separation to 0-1 scale
- Caps at 1.0 for separations ≥ 10 yards
- If `sep_nearest = ∞`, then `sep_score = 0.5` (neutral)

### 4.2 Speed Score
```
speed_score = min(receiver_speed / 8.0, 1.0)
```
- Normalizes speed to 0-1 scale
- Caps at 1.0 for speeds ≥ 8 yards/second
- If `receiver_speed ≤ 0`, then `speed_score = 0.0`

### 4.3 Acceleration Score
```
accel_score = (min(max(receiver_accel / 3.0, -1.0), 1.0) / 2.0) + 0.5
```
- Clamps acceleration to [-1, 1] range (normalized by 3.0 yds/s²)
- Shifts to [0, 1] range by dividing by 2 and adding 0.5
- Positive acceleration → higher score, negative → lower score

### 4.4 Time Score
```
if time_advantage is finite:
    time_score = (min(max(time_advantage / 2.0, -1.0), 1.0) / 2.0) + 0.5
else:
    time_score = 0.5
```
- Clamps time advantage to [-1, 1] range (normalized by 2.0 seconds)
- Shifts to [0, 1] range
- Positive time advantage → higher score

### 4.5 Pressure Score
```
pressure_score = 1.0 - min(num_def_within_3 / 5.0, 1.0)
```
- Inverse of defender pressure
- More defenders within 3 yards → lower score
- Caps at 0.0 when 5+ defenders within 3 yards

### 4.6 Leverage Score
```
leverage_score = leverage_angle / 180.0
```
- Normalizes leverage angle to 0-1 scale
- 180° angle (defender directly in front) → score = 1.0
- 0° angle (defender directly behind) → score = 0.0
- If leverage angle is NaN, then `leverage_score = 0.5` (neutral)

**Leverage Angle Calculation:**
```
# Vector from defender to receiver
def_to_rec_x = receiver_x - defender_x
def_to_rec_y = receiver_y - defender_y

# Vector from receiver to ball
rec_to_ball_x = ball_land_x - receiver_x
rec_to_ball_y = ball_land_y - receiver_y

# Dot product
dot_product = def_to_rec_x × rec_to_ball_x + def_to_rec_y × rec_to_ball_y

# Magnitudes
mag_def_to_rec = √(def_to_rec_x² + def_to_rec_y²)
mag_rec_to_ball = √(rec_to_ball_x² + rec_to_ball_y²)

# Angle in radians
cos_angle = dot_product / (mag_def_to_rec × mag_rec_to_ball)
cos_angle = clip(cos_angle, -1.0, 1.0)  # Prevent numerical errors
angle_rad = arccos(cos_angle)
angle_deg = radians_to_degrees(angle_rad)

# Normalize to [0, 180°]
leverage_angle = min(angle_deg, 180.0 - angle_deg)
```

---

## Summary

| Metric | Formula | Range | Higher is Better? |
|--------|---------|-------|-------------------|
| **Time to Ball** | `distance / speed_toward_ball` | [0, ∞] seconds | No (lower is better) |
| **Defender Time to Ball** | `def_distance / def_speed_toward_ball` | [0, ∞] seconds | No (lower is better) |
| **Time Advantage** | `def_time_to_ball - time_to_ball` | (-∞, ∞) seconds | Yes |
| **Dominance Score** | Weighted sum of 6 components | [0, 1] | Yes |

**Dominance Score Weights:**
- Separation: 25%
- Speed: 18%
- Acceleration: 12%
- Time Advantage: 18%
- Pressure: 12%
- Leverage: 15%

---

## Notes

1. **Coordinate System**: All coordinates are standardized so offense always drives right (left-to-right orientation).

2. **Units**:
   - Distances: yards
   - Speeds: yards/second
   - Accelerations: yards/second²
   - Times: seconds
   - Angles: degrees

3. **Frame Rate**: NFL tracking data uses 10 frames per second (`FRAME_RATE = 10`).

4. **Ball Landing Position**: The ball landing position (`ball_land_x`, `ball_land_y`) is determined from the output data (post-throw frames).

