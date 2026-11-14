# Paired physics thermal and RGB cameras

The Missions dashboard now saves an atomic thermal/RGB pair at each observation.
The side-by-side viewer also offers separate Thermal and RGB tabs. The agent
receives thermal first, RGB second; its boxes refer to thermal pixels only.

RGB is a native Gazebo camera over a photographic ground material. Thermal is
a perspective CPU sensor driven by the same native odometry, timestamp, field
of view and camera mount. It samples an experimental temperature surface; it
does not replay a flight frame and is not Gazebo's old purple material view.
The native RGB shadow need not appear in thermal: transient moving-shadow
heating and three-dimensional thermal occlusion are not modeled.

The temperature surface uses our existing DuMux target-present and target-absent
24-hour surface exports. The difference is inserted at its original 1m domain
scale at the buried-target locations. A periodic semi-infinite soil heat model
adds material variation with surface admittance
`h + (1+i)*sqrt(k*C*omega/2)`. Assigned albedo follows the RGB diffuse material;
recorded thermal training crops supply low-amplitude empirical microtexture.
The thermal display uses white-hot percentile AGC, nominal 0.04K noise and
0.65-pixel blur. These are sensor assumptions, not measured calibration.

The UI offers Warming (DuMux time 7.5h, maximum inserted contrast about +2.49K)
and Cooling (time 17h, minimum contrast about −1.09K). These are fixed physics
snapshots. RGB illumination is held constant to provide comparable context.

## Scope and quality limits

This is an experimental composition of small DuMux solutions plus a reduced
ground-response model. It is **not** a full 70m coupled PDE solve. The existing
8×8 surface cases have not passed all numerical convergence and field gates;
interpolated display pixels do not add physical solver resolution. These
demonstration scenes are not admitted to the validated training stream.

RGB material is [Aerial Grass Rock by Rob Tuytel](https://polyhaven.com/a/aerial_grass_rock),
[CC0](https://polyhaven.com/license), with its documented 15m source scale.
It is separate from our recorded RGB: the local samples checked were indoor
or obscured. No real-flight RGB/thermal registration is claimed.

## Verification and artifacts

- The white-hot output is grayscale with many intensity levels, rather than
  the old coarse purple palette.
- An actual hover tested the eight rotation/reflection alternatives against
  native RGB terrain. The intended mapping had correlation−0.853 for the
  assigned warmer-dark-ground response; alternatives were much weaker. This
  is a geometry/material consistency check, not thermal field validation.
- Captured pair IDs, image hashes, sensor timestamps and native poses must
  agree. Incorrect pair time/pose and changed field bytes are rejected.
- The paired-image Gemini agent completed run `ca1f8d19b1b64ab3` with a lower
  revisit; cooling reference run `d00accbd508d4baa` also completed.

Source generation: `simulation/prepare_sensor_scene.py --physics-hour 7.5`,
then `--physics-hour 17` for the alternate snapshot. Generate the warming/cooling
manifest files before starting the adapter. Immutable temperature fields and
their sidecars live on the external SSD under `simulation/sensor-scene/fields`.
The server and API remain persistent user services.

The presentation figure uses an actual captured RGB/warming pair and renders
the cooling field at that exact native camera pose. It is labeled as a
controlled fixed-pose comparison, not a second live flight capture.
