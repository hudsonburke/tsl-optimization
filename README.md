# tsl-optimization

Refined from poster presented at ASB 2025 Conference: [Link](https://github.com/hudsonburke/ASB2025-TSL-Optimization)

## What this package does

This package estimates muscle fiber lengths and tendon slack length from sampled muscle-tendon lengths and force-length curves.

Public entry points:

- `optimize_fiber_length(...)` → optimize a fiber length at each sampled pose.
- `optimize_fiber_length_and_tsl(...)` → jointly optimize pose-specific fiber lengths and one shared tendon slack length.
- `calc_tsl(...)` → evaluate tendon slack length from a fixed set of fiber lengths.
- `calc_pennation(...)` → compute pennation angle from normalized fiber length.
- `CurveWrapper` / `evaluate_curve(...)` → evaluate NumPy or OpenSim-style curves through one interface.

## Current approach

### 1. Curve handling

- Curves can be provided either as:
  - a NumPy array with shape `(2, n)`, where row 0 is x and row 1 is y, or
  - an OpenSim-like object exposing `calcValue(float) -> float`.
- `CurveWrapper` prebuilds SciPy interpolants for forward evaluation.
- For inverse evaluation, the curve must be:
  - strictly monotonic, or
  - nondecreasing with a leading plateau followed by a strictly increasing segment.
- Genuinely non-monotonic curves do not get an inverse.

This is used for the tendon force-length curve, where inverse evaluation maps normalized tendon force back to normalized tendon length.

### 2. Muscle/tendon model

For each sampled pose:

1. Normalize fiber length: `lm_norm = lm / lm_opt`.
2. Compute pennation with `calc_pennation(...)`.
3. Compute normalized muscle force from active + passive force-length curves.
4. Project muscle force onto the tendon with `cos(alpha)`.
5. Invert the tendon force-length curve to get normalized tendon length `lt_norm`.
6. Compute geometric tendon length from the sampled musculotendon length:

   `lt_geom = lmt - lm * cos(alpha)`

7. Convert geometric tendon length and normalized tendon length into tendon slack length.

Important behavior:

- Tendon compression is not allowed. Negative projected tendon force is clamped to zero before inverse tendon evaluation.
- Zero/near-zero tendon force maps to normalized tendon length `1.0` (slack).
- `calc_tsl(strict=True)` raises on nonphysical tendon states instead of silently sanitizing them.
- `calc_tsl(strict=False, clip=True)` keeps the previous user-facing behavior by clipping slack length into physical bounds.

### 3. Optimization strategy

#### `optimize_fiber_length(...)`

- Decision variables: one fiber length per sampled pose.
- Objective: minimize a scalar objective over the implied tendon slack lengths.
- Default objective: `ssdp`, the sum of squared pairwise differences, which favors a consistent slack length across poses.
- Supported solvers: constrained methods only (`SLSQP`, `trust-constr`). Default: `SLSQP`.

#### `optimize_fiber_length_and_tsl(...)`

- Decision variables: one fiber length per sampled pose plus one shared tendon slack length.
- Objective: minimize the residual between:
  - geometric tendon length from kinematics, and
  - tendon length implied by force equilibrium and the shared slack length.

### 4. Hard no-buckling constraint

The current implementation treats tendon buckling prevention as a hard constraint.

For `optimize_fiber_length(...)`:

- geometric tendon length must remain nonnegative at every sampled pose.

For `optimize_fiber_length_and_tsl(...)`:

- geometric tendon length must satisfy

  `lmt - lm * cos(alpha) >= (1 + tendon_margin) * lts`

  at every sampled pose.

This means:

- unconstrained optimizers such as `L-BFGS-B` are intentionally rejected,
- the objective does not "buy" a better fit by allowing buckling,
- and returned solutions are checked again after the solver reports success.

## Development

- Create/update the environment: `uv sync --group dev`
- Run the test suite: `uv run pytest -q`

## Notes

- NumPy workflows require only the declared `uv` dependencies.
- OpenSim is optional, but OpenSim-backed curves require an installed runtime that provides `calcValue()`-compatible objects.
- The project uses a `src/` layout and is configured through `pyproject.toml`.
