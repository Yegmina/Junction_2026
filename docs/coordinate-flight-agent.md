# Coordinate tools and live investigation

The live agent uses the `coordinate-flight-v3` protocol. Its tools are
`goto`, `observe`, `hover`, and `return_home`. `goto` accepts local ENU metres:
`x_m` east, `y_m` north, `altitude_m` above the flat ground, and optional
`yaw_deg`. Destinations can be anywhere inside the ±24m horizontal geofence
and 2–10m altitude envelope. Named regions are landmarks, not the action space.

The mission manager sends the model synchronized thermal/RGB evidence,
capture pose, current navigation feedback, recent tool results and remaining
simulated flight time. The model returns a coordinate command plus concise
observation, decision-rationale and next-check summaries. Those summaries are
shown alongside the native camera with actual tool arguments and feedback.
They are operational explanations, not a raw internal reasoning transcript.

The world continues during inference. Telemetry and route records update while
the request is pending. A goto is acknowledged immediately, followed by a
camera check after 0.5–5 simulated seconds; the next goto can retarget an
aircraft that is still moving. `observe` keeps the active destination and
`hover` commands a hold. Stops and the estimated home-return reserve cancel
pending inference and initiate a return. Flight time during inference counts
against the budget. No flight progress is fabricated to hide model latency.

Coordinates are checked before dispatch. Declared excluded regions apply to
the full straight-line segment. Rejected calls are visible and returned as
feedback for correction; three consecutive failed calls trigger recovery.
Automatic return can route around declared rectangular exclusions. The flight
controller continues to use native velocity control and integrated Gazebo
multirotor dynamics. This interface has no real-aircraft connection.

Paired observations keep their capture identities. Detection boxes apply to
the frame sent with that decision, even as the aircraft moves. Raw requests and
responses are saved with the new prompt version and unique step identifiers.
The Reference controller retains its scripted region policy. Existing legacy
area-action GPU probes do not validate this new coordinate-tool protocol or
establish that a trained checkpoint improves autonomous navigation.

## Deployment and verification — 14 November 2026

The public VM uses `/opt/aerorozum/deployment/releases/coordinate-agent-v3`.
Only the API service was restarted; the simulator, camera adapter, inference
server and ongoing physics jobs were preserved. The previous release remains
available for rollback.

All 261 Python tests pass and the TypeScript/Vite production build passes.
Public-browser checks passed at desktop and mobile widths without JavaScript
errors or mobile horizontal overflow. Fresh visits still obtain automatic guest
sessions without a password prompt.

Native Gazebo mission `24859da319234e7d`, using the configured Gemini agent,
issued `goto(22,22,4)` and then `goto(6,-6,3)` while still moving toward the first
target. Actual telemetry progressed during the intervening model request. The
second command records `adjusted_during_motion: true`; the first target was
still approximately 21m away when the new destination was dispatched. This
checks the coordinate tool mechanism and live retargeting, not autonomous
detection quality or trained-policy improvement.

The four real model decisions were `goto → goto → observe → return_home`.
The mission completed without an error and confirmed home return after 48.11
simulated seconds, within its 120-second budget. Model input/response logs and
camera evidence are archived privately under
`gs://aerorozum-thermal-rl-avid-relic-509408-e4/mission-demos/coordinate-flight-v3/24859da319234e7d`.

Presentation assets are saved under `artifacts/presentation/`:
`coordinate-agent-presentation.png`, `coordinate-agent-mobile.png`, and
`coordinate-agent-tools.mp4`. The console contains concise observation and
operational decision summaries, exact tool arguments, and actual feedback.
