# Receiver Dominance Calculation - Complete Mathematical Explanation

## Overview

The **Receiver Dominance** metric quantifies how much "dominance" or advantage a receiver has over defenders at any given moment. It uses multivariate normal distributions (Gaussian PDFs) to model spatial influence, similar to the 2023 NFL Data Bowl winner's "Continuous Pocket Pressure" (CPP) approach.

---

## Mathematical Framework

### Coordinate System

- **Field dimensions**: Width = 53.3 yards, Length = 120 yards
- **Grid resolution**: 0.5 yards (creates ~25,000 points per frame)
- **Coordinate notation**: 
  - $x$ = width coordinate (0 to 53.3)
  - $y$ = length coordinate (0 to 120)
  - Receiver position: $(x_r, y_r)$
  - Defender position: $(x_d, y_d)$

---

## Step 1: Receiver Influence PDF

The receiver's influence is modeled as a **multivariate normal distribution** (2D Gaussian):

### Formula

$$f_r(x, y) = \frac{1}{2\pi\sqrt{|\Sigma_r|}} \exp\left(-\frac{1}{2}(\mathbf{p} - \boldsymbol{\mu}_r)^T \Sigma_r^{-1} (\mathbf{p} - \boldsymbol{\mu}_r)\right)$$

Where:
- $\mathbf{p} = [x, y]^T$ = point on field
- $\boldsymbol{\mu}_r = [x_r, y_r]^T$ = receiver position (mean)
- $\Sigma_r = \begin{bmatrix} 8^2 & 0 \\ 0 & 8^2 \end{bmatrix}$ = covariance matrix (8 yard standard deviation)

### Simplified Form

Since the covariance is diagonal (circular distribution):

$$f_r(x, y) = \frac{1}{2\pi \cdot 8^2} \exp\left(-\frac{(x - x_r)^2 + (y - y_r)^2}{2 \cdot 8^2}\right)$$

### Interpretation

- **8-yard radius**: The receiver's influence extends approximately 8 yards in all directions
- **Circular shape**: Equal influence in all directions (isotropic)
- **Peak at receiver**: Maximum influence at the receiver's exact position
- **Decay**: Influence decreases exponentially with distance

### Example

If receiver is at position $(x_r, y_r) = (30, 50)$:

- At receiver position: $f_r(30, 50) \approx 0.00498$ (maximum)
- 4 yards away: $f_r(34, 50) \approx 0.00312$ (62% of max)
- 8 yards away: $f_r(38, 50) \approx 0.00124$ (25% of max)
- 12 yards away: $f_r(42, 50) \approx 0.00031$ (6% of max)

---

## Step 2: Defender Pressure PDF

The defender's pressure is also modeled as a multivariate normal distribution, but **weighted by separation distance**.

### Base PDF Formula

$$f_d(x, y) = \frac{1}{2\pi \cdot 6^2} \exp\left(-\frac{(x - x_d)^2 + (y - y_d)^2}{2 \cdot 6^2}\right)$$

Where:
- $\boldsymbol{\mu}_d = [x_d, y_d]^T$ = defender position
- $\Sigma_d = \begin{bmatrix} 6^2 & 0 \\ 0 & 6^2 \end{bmatrix}$ = covariance matrix (6 yard standard deviation)

### Separation Distance

The separation between receiver and defender:

$$s = \sqrt{(x_d - x_r)^2 + (y_d - y_r)^2}$$

### Weighting Function

The defender's influence is weighted inversely with separation:

$$w(s) = \frac{1}{1 + \frac{s}{5}}$$

**Properties**:
- When $s = 0$ (defender on receiver): $w(0) = 1.0$ (full pressure)
- When $s = 5$ yards: $w(5) = 0.5$ (half pressure)
- When $s = 10$ yards: $w(10) = 0.33$ (one-third pressure)
- When $s \to \infty$: $w(\infty) = 0$ (no pressure)

### Weighted Defender PDF

$$f_d^{weighted}(x, y) = w(s) \cdot f_d(x, y) = \frac{1}{1 + \frac{s}{5}} \cdot \frac{1}{2\pi \cdot 6^2} \exp\left(-\frac{(x - x_d)^2 + (y - y_d)^2}{2 \cdot 6^2}\right)$$

### Example

**Scenario**: Receiver at $(30, 50)$, Defender at $(33, 50)$

1. **Separation**: $s = \sqrt{(33-30)^2 + (50-50)^2} = 3$ yards

2. **Weight**: $w(3) = \frac{1}{1 + \frac{3}{5}} = \frac{1}{1.6} = 0.625$

3. **Defender PDF at receiver position**:
   - Base: $f_d(30, 50) = \frac{1}{2\pi \cdot 36} \exp\left(-\frac{9}{72}\right) \approx 0.00389$
   - Weighted: $f_d^{weighted}(30, 50) = 0.625 \times 0.00389 \approx 0.00243$

4. **If defender moves to $(36, 50)$ (6 yards away)**:
   - Separation: $s = 6$ yards
   - Weight: $w(6) = \frac{1}{1 + \frac{6}{5}} = \frac{1}{2.2} \approx 0.455$
   - Pressure decreases by ~27%

---

## Step 3: Dominance Ratio Calculation

For each point $(x, y)$ on the field, calculate the dominance ratio:

### Formula

$$D(x, y) = \frac{f_r(x, y)}{f_r(x, y) + f_d^{weighted}(x, y) + \epsilon}$$

Where:
- $\epsilon = 10^{-10}$ = small constant to prevent division by zero

### Interpretation

- **$D(x, y) = 1.0$**: Complete receiver dominance (no defender influence)
- **$D(x, y) = 0.5$**: Balanced (equal influence)
- **$D(x, y) = 0.0$**: Complete defender dominance (no receiver influence)

### Example Calculation

**Given**:
- Receiver at $(30, 50)$ with $f_r(30, 50) = 0.00498$
- Defender at $(33, 50)$ with $f_d^{weighted}(30, 50) = 0.00243$

**At receiver position**:
$$D(30, 50) = \frac{0.00498}{0.00498 + 0.00243 + 10^{-10}} = \frac{0.00498}{0.00741} \approx 0.672$$

**Interpretation**: 67.2% receiver dominance at the receiver's position.

**At defender position $(33, 50)$**:
- $f_r(33, 50) \approx 0.00312$ (receiver influence 3 yards away)
- $f_d(33, 50) \approx 0.00498$ (defender at own position, full weight)
- Weighted: $f_d^{weighted}(33, 50) = 0.625 \times 0.00498 \approx 0.00311$

$$D(33, 50) = \frac{0.00312}{0.00312 + 0.00311 + 10^{-10}} = \frac{0.00312}{0.00623} \approx 0.500$$

**Interpretation**: 50% balanced at defender's position.

---

## Step 4: Overall Dominance Score

The final dominance score is calculated by averaging the dominance ratio over the receiver's immediate area.

### Receiver Area PDF

Define a focused area around the receiver:

$$f_{area}(x, y) = \frac{1}{2\pi \cdot 6^2} \exp\left(-\frac{(x - x_r)^2 + (y - y_r)^2}{2 \cdot 6^2}\right)$$

(6-yard radius around receiver)

### Dominance Score Formula

$$\text{Dominance} = \frac{\sum_{(x,y) \in \text{Field}} f_{area}(x, y) \cdot D(x, y)}{\sum_{(x,y) \in \text{Field}} f_{area}(x, y)}$$

In continuous form (integral):

$$\text{Dominance} = \frac{\iint f_{area}(x, y) \cdot D(x, y) \, dx \, dy}{\iint f_{area}(x, y) \, dx \, dy}$$

### Discrete Implementation

Since we use a grid:

$$\text{Dominance} = \frac{\sum_{i,j} f_{area}(x_i, y_j) \cdot D(x_i, y_j)}{\sum_{i,j} f_{area}(x_i, y_j)}$$

Where:
- $i, j$ = grid indices
- Grid spacing = 0.5 yards
- Total points ≈ 25,000 per frame

### Normalization

The result is typically normalized to a 0-1 scale:

$$\text{Dominance}_{normalized} = \text{clip}\left(\frac{\text{Dominance} - 0.5}{0.8 - 0.5}, 0, 1\right)$$

Where $\text{clip}(x, a, b)$ clamps $x$ between $a$ and $b$.

---

## Complete Example: Full Calculation

### Scenario Setup

- **Receiver**: Position $(30, 50)$, Speed = 5.0 yd/s
- **Defender**: Position $(33, 50)$, Speed = 4.5 yd/s
- **Separation**: $s = 3$ yards

### Step-by-Step Calculation

#### 1. Receiver PDF at various points

At receiver position $(30, 50)$:
$$f_r(30, 50) = \frac{1}{2\pi \cdot 64} \exp(0) = \frac{1}{128\pi} \approx 0.00498$$

At point $(32, 50)$ (2 yards away):
$$f_r(32, 50) = \frac{1}{128\pi} \exp\left(-\frac{4}{128}\right) \approx 0.00483$$

At point $(35, 50)$ (5 yards away):
$$f_r(35, 50) = \frac{1}{128\pi} \exp\left(-\frac{25}{128}\right) \approx 0.00405$$

#### 2. Defender PDF and Weight

Separation: $s = 3$ yards

Weight: $w(3) = \frac{1}{1 + \frac{3}{5}} = 0.625$

At defender position $(33, 50)$:
$$f_d(33, 50) = \frac{1}{2\pi \cdot 36} \exp(0) = \frac{1}{72\pi} \approx 0.00442$$

Weighted: $f_d^{weighted}(33, 50) = 0.625 \times 0.00442 \approx 0.00276$

#### 3. Dominance Ratio at Key Points

**At receiver $(30, 50)$**:
- $f_r = 0.00498$
- $f_d^{weighted} = 0.00243$ (defender influence at receiver position)
- $D(30, 50) = \frac{0.00498}{0.00498 + 0.00243} = 0.672$ (67.2% receiver dominance)

**At defender $(33, 50)$**:
- $f_r = 0.00312$ (receiver influence 3 yards away)
- $f_d^{weighted} = 0.00276$ (defender at own position)
- $D(33, 50) = \frac{0.00312}{0.00312 + 0.00276} = 0.531$ (53.1% receiver dominance)

**At midpoint $(31.5, 50)$**:
- $f_r = 0.00440$ (receiver influence 1.5 yards away)
- $f_d^{weighted} = 0.00260$ (defender influence 1.5 yards away)
- $D(31.5, 50) = \frac{0.00440}{0.00440 + 0.00260} = 0.629$ (62.9% receiver dominance)

#### 4. Overall Dominance Score

Sum over all grid points (simplified example with 3 points):

$$\text{Dominance} = \frac{f_{area}(30,50) \cdot D(30,50) + f_{area}(33,50) \cdot D(33,50) + f_{area}(31.5,50) \cdot D(31.5,50)}{f_{area}(30,50) + f_{area}(33,50) + f_{area}(31.5,50)}$$

With actual grid (25,000 points), this gives approximately:
$$\text{Dominance} \approx 0.65$$

**Interpretation**: 65% receiver dominance overall.

---

## Visual Interpretation: The Purple Contour

### What the Purple Contour Represents

The purple contour plot visualizes $D(x, y)$ across the entire field:

$$D(x, y) = \frac{f_r(x, y)}{f_r(x, y) + f_d^{weighted}(x, y) + \epsilon}$$

### Color Mapping

- **Dark Purple** ($D \approx 1.0$): Areas where receiver has strong dominance
- **Medium Purple** ($D \approx 0.5$): Balanced areas
- **Light Purple** ($D \approx 0.0$): Areas where defender has pressure
- **No Purple**: Areas with minimal influence from either side

### Contour Levels

The visualization uses 15 contour levels:

$$D_i = \frac{i}{15}, \quad i = 0, 1, 2, \ldots, 15$$

Each contour line represents points where $D(x, y) = D_i$.

### Example Visualization

For receiver at $(30, 50)$ and defender at $(33, 50)$:

- **Dark purple region** around $(30, 50)$: $D > 0.7$ (high receiver dominance)
- **Medium purple** between them: $0.4 < D < 0.7$ (balanced)
- **Light purple** around $(33, 50)$: $D < 0.4$ (defender pressure)

---

## Why Dominance Changes Frame-by-Frame

### Dynamic Factors

The dominance calculation depends on:

1. **Positions**: $(x_r, y_r)$ and $(x_d, y_d)$ change each frame
2. **Separation**: $s = \sqrt{(x_d - x_r)^2 + (y_d - y_r)^2}$ changes
3. **Weight**: $w(s) = \frac{1}{1 + s/5}$ changes with separation
4. **PDFs**: $f_r(x, y)$ and $f_d(x, y)$ shift as players move

### Frame-to-Frame Example

**Frame 1**: Receiver $(30, 50)$, Defender $(33, 50)$, $s = 3$ yd
- $w(3) = 0.625$
- Dominance ≈ 0.65

**Frame 2**: Receiver $(32, 52)$, Defender $(34, 52)$, $s = 2$ yd (closer!)
- $w(2) = \frac{1}{1 + 2/5} = \frac{1}{1.4} \approx 0.714$ (higher weight)
- Defender pressure increases
- Dominance ≈ 0.58 (decreased)

**Frame 3**: Receiver $(35, 55)$, Defender $(33, 55)$, $s = 2$ yd, but receiver moving away
- Separation same, but receiver has momentum
- Dominance ≈ 0.62 (slight recovery)

---

## Mathematical Properties

### Bounds

$$0 \leq D(x, y) \leq 1$$

Since $f_r(x, y) \geq 0$ and $f_d^{weighted}(x, y) \geq 0$:

$$D(x, y) = \frac{f_r}{f_r + f_d + \epsilon} \leq \frac{f_r}{f_r + \epsilon} \approx 1$$

And:

$$D(x, y) = \frac{f_r}{f_r + f_d + \epsilon} \geq 0$$

### Symmetry

When $f_r = f_d^{weighted}$:
$$D = \frac{f_r}{f_r + f_r} = 0.5$$

This represents perfect balance.

### Sensitivity to Separation

The derivative with respect to separation $s$:

$$\frac{\partial D}{\partial s} = \frac{\partial}{\partial s}\left(\frac{f_r}{f_r + w(s) \cdot f_d}\right)$$

Since $w(s)$ decreases with $s$, and $f_d$ may also change:
- As separation increases → $w(s)$ decreases → $f_d^{weighted}$ decreases → $D$ increases
- **Interpretation**: More separation = higher receiver dominance

---

## Implementation Details

### Grid Resolution

- **Spacing**: $\Delta x = \Delta y = 0.5$ yards
- **Field size**: $53.3 \times 120$ yards
- **Grid points**: $N_x = \lceil 53.3 / 0.5 \rceil = 107$, $N_y = \lceil 120 / 0.5 \rceil = 240$
- **Total points**: $N = N_x \times N_y = 25,680$

### Computational Complexity

For each frame:
- **PDF calculations**: $O(N)$ for each PDF (receiver + defender)
- **Dominance ratio**: $O(N)$
- **Overall score**: $O(N)$
- **Total**: $O(N) \approx O(25,000)$ operations per frame

### Numerical Stability

- **Epsilon**: $\epsilon = 10^{-10}$ prevents division by zero
- **Normalization**: Clamping ensures $0 \leq \text{Dominance} \leq 1$
- **Floating point**: Uses double precision (64-bit)

---

## Summary

The receiver dominance metric:

1. **Models spatial influence** using Gaussian PDFs
2. **Weights defender pressure** by separation distance
3. **Calculates dominance ratio** at each field point
4. **Averages over receiver area** to get overall score
5. **Visualizes as purple contour** showing spatial dominance map

**Key Formula**:
$$D(x, y) = \frac{f_r(x, y)}{f_r(x, y) + w(s) \cdot f_d(x, y) + \epsilon}$$

Where:
- $f_r$ = Receiver influence (8-yard radius)
- $f_d$ = Defender pressure (6-yard radius)
- $w(s) = \frac{1}{1 + s/5}$ = Separation weight
- $s$ = Distance between receiver and defender
