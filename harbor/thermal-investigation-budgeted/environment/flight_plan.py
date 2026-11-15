"""Public waypoint executor: only documented tools; no detection or scene access."""
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def validate(plan):
    points = plan.get('waypoints')
    if not isinstance(points, list) or not 1 <= len(points) <= 16:
        raise ValueError('waypoints must contain 1..16 objects')
    clean = []
    for point in points:
        values = [point.get(k) for k in ('x', 'y', 'altitude')]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise ValueError('Every waypoint needs finite x, y and altitude')
        x, y, z = values
        if not (-24 <= x <= 24 and -24 <= y <= 24 and 2 <= z <= 10):
            raise ValueError('Waypoint outside public flight bounds')
        clean.append(dict(x=x, y=y, altitude=z))
    return clean


def home_reserve(state):
    p = state['pose']
    # Conservative flight estimate plus settling and command overhead.
    return math.hypot(p['x_m'], p['y_m']) / 1.2 + max(2, p['z_m']) / .8 + 6


def collect(plan, output=Path('/workspace/evidence/mission.json'), *, call=None, sleep=time.sleep):
    points = validate(plan)  # Validate everything before a launch can occur.
    output = Path(output)
    base = os.environ.get('DRONE_API', 'http://simulator:8080')
    if call is None:
        def call(body):
            request = urllib.request.Request(base + '/tool', data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=240) as response:
                return json.load(response)
    record = dict(plan=dict(waypoints=points), observations=[], events=[], errors=[],
                  status='preflight', public_tools_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(output)

    def event(body):
        result = call(body)
        record['events'].append(dict(command=body, status=result.get('status', result)))
        save()
        return result

    initial = event(dict(command='status'))
    if initial['flight_elapsed_s'] or initial.get('report_submitted'):
        raise ValueError('collect requires a fresh mission; it cannot reset a flight')
    record['status'] = 'flying'
    launched = False
    try:
        launched = True  # Also attempt return if a launch request times out.
        event(dict(command='launch', altitude=2))
        for point in points:
            state = event(dict(command='status'))
            p = state['pose']
            travel = max(math.hypot(point['x']-p['x_m'], point['y']-p['y_m']) / 1.2,
                         abs(point['altitude']-p['z_m']) / .8) + 3
            destination = dict(pose=dict(x_m=point['x'], y_m=point['y'], z_m=point['altitude']))
            if state['remaining_s'] < travel + home_reserve(destination) + 2:
                record['events'].append(dict(skipped_waypoint=point, reason='return reserve'))
                break
            if state['action_count'] >= 56:
                break
            event(dict(command='goto', **point))
            deadline = time.monotonic() + 90
            arrived = False
            while time.monotonic() < deadline:
                state = call(dict(command='status'))
                p = state['pose']
                if state['remaining_s'] <= home_reserve(state) + 2:
                    break
                if math.dist((p['x_m'], p['y_m'], p['z_m']),
                             (point['x'], point['y'], point['altitude'])) < .4:
                    arrived = True
                    break
                sleep(.15)
            if not arrived:
                break
            for attempt in range(3):
                try:
                    observation = event(dict(command='observe'))
                    record['observations'].append(observation)
                    save()
                    break
                except urllib.error.HTTPError as error:
                    if error.code != 400 or attempt == 2:
                        raise
                    state = call(dict(command='status'))
                    if state['remaining_s'] <= home_reserve(state) + 3:
                        break
                    sleep(.2)
    except Exception as error:
        # Evidence survives an agent-side command failure; do not submit guesses.
        record['errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        if launched:
            try:
                record['landing'] = event(dict(command='land'))
                record['status'] = 'landed'
            except Exception as error:
                record['errors'].append('Return failed: ' + str(error))
                record['status'] = 'return_failed'
        save()
    # Download after landing: network/image I/O must not spend flight budget.
    for observation in record['observations']:
        for item in observation.get('images', []):
            try:
                destination = output.parent / Path(item['filename']).name
                with urllib.request.urlopen(base + item['url'], timeout=30) as response:
                    destination.write_bytes(response.read())
                item['local_file'] = str(destination)
            except Exception as error:
                record['errors'].append('Image download: ' + str(error))
    save()
    return dict(status=record['status'], index_file=str(output),
                observations=len(record['observations']), landing=record.get('landing'), errors=record['errors'])
