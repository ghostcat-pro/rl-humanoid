# INSPIRATION

Ideas to borrow from the external humanoid project for future work.

## Robustness & Randomization
- Add domain randomization: friction range, base mass perturbation, periodic pushes (linear + angular), action delay and multiplicative noise.
- Include indicators in observations for randomization (e.g., friction, mass) so the critic can adapt.

## Gait & Phase Awareness
- Derive a simple gait phase from elapsed time; use stance/swing masks.
- Build phase-based reference joint targets and optionally blend reference actions into policy output (anneal out later).
- Use phase/stance masks in rewards (expected contacts, swing clearance, air time).

## Observations
- Asymmetric stacking: more frames for policy obs, fewer for privileged obs to reduce size.
- Privileged extras: ref-action diffs, stance/contact masks, friction/mass noise indicators, push vectors, phase (sin/cos), base kinematics.
- Per-channel noise scaling: add noise to joints/vel/ang/quat, skip commands/history.

## Action & Control Hygiene
- Lower action scale with PD control; clip actions.
- Apply randomized delay blending with previous actions plus mild multiplicative noise to smooth torques.

## Reward Shaping Candidates
- Foot slip penalty (contact × foot velocity norm).
- Feet/knee distance band (not too narrow/wide).
- Feet air-time bonus for longer swing steps; swing-foot clearance to target height.
- Contact pattern vs expected gait phase (reward when contact matches stance).
- Base height tracking using stance-foot height reference.
- Orientation via Euler deviation and projected gravity.
- Velocity mismatch suppression on unused axes (lin-z, ang x/y).
- “Track vel hard”: exp tracking plus linear penalty on error magnitude.
- Action smoothness (Δa and Δ²a) beyond torque/vel costs.
- Low-speed penalty when under target speed or sign-mismatched to command.

## Commands & Heading
- Heading-based yaw commands with moderate resampling window to stabilize turning.

## PPO/Model Hints
- Larger nets for harder tasks (e.g., actor 512-256-128, critic 768-256-128).
- Try lower LR and fewer epochs if seeing instability/overfit; modest entropy (~0.001).
