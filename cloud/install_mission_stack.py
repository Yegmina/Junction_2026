"""Install a prepared release as boot-persistent, loopback-only VM services."""
import argparse
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
from datetime import datetime, timedelta, timezone


def run(*args):
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/opt/aerorozum/deployment'))
    parser.add_argument('--provider', choices=['gemini_api', 'local_gemma'], default='gemini_api')
    parser.add_argument('--stop-at', help='Fixed ISO UTC deployment deadline from the VM control plane')
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit('Run as root on the VM')
    release = args.release.resolve()
    root = args.root.resolve()
    sim = root/'simulation'
    data = root/'artifacts'
    if not (release/'frontend/dist/index.html').exists() or not (sim/'worlds/aerorozum.sdf').exists():
        raise SystemExit('Transfer the built dashboard and simulation assets first')
    try:
        user = pwd.getpwnam('aerorozum-app')
    except KeyError:
        run('useradd', '--system', '--home-dir', str(root/'simulation/runtime'), '--shell', '/usr/sbin/nologin', 'aerorozum-app')
        user = pwd.getpwnam('aerorozum-app')
    if user.pw_dir == str(root):
        run('usermod', '--home', str(root/'simulation/runtime'), 'aerorozum-app')
    for name in ['artifacts', 'cache', 'logs', 'simulation/media', 'simulation/runtime',
                 'simulation/cache', 'simulation/config', 'simulation/logs', 'simulation/fuel']:
        folder = root/name
        folder.mkdir(parents=True, exist_ok=True)
        shutil.chown(folder, user=user.pw_uid, group=user.pw_gid)
    # Only this deployment's runtime state is writable by the application.
    run('chown', '-R', 'aerorozum-app:aerorozum-app', str(data), str(sim/'media'))
    original = json.loads((sim/'deployment-source.json').read_text())['simulation_root']
    for path in [sim/'worlds/aerorozum.sdf', *list((sim/'sensor-scene').glob('*.json'))]:
        path.write_text(path.read_text().replace(original, str(sim)))
    venv = root/'venv'
    if not (venv/'bin/python').exists():
        run('/usr/bin/python3', '-m', 'venv', str(venv))
    run(str(venv/'bin/pip'), 'install', '--disable-pip-version-check', str(release))
    # Uvicorn runs the source checkout so the bundled dashboard is served too.
    env = {
        'AEROROZUM_SIM_ROOT': str(sim), 'AEROROZUM_DATA_ROOT': str(data),
        'AEROROZUM_MISSION_PROVIDER': args.provider,
        'AEROROZUM_MISSION_MODEL': 'google/gemma-4-12B-it/real_sft',
        'GZ_PARTITION': 'aerorozum-mission-local', 'GZ_IP': '127.0.0.1',
        'GZ_DISCOVERY_MSG_PORT': '18935', 'GZ_DISCOVERY_SRV_PORT': '18936',
        'GZ_SIM_RESOURCE_PATH': str(sim/'models'), 'GZ_FUEL_CACHE_PATH': str(sim/'fuel'),
        'GZ_SIM_LOG_ROOT': str(sim/'logs'), 'GZ_LOG_PATH': str(sim/'logs'),
        'XDG_CACHE_HOME': str(sim/'cache'), 'XDG_CONFIG_HOME': str(sim/'config'),
        # A100 has no graphics engine. Render native Gazebo RGB with Mesa CPU EGL.
        'LIBGL_ALWAYS_SOFTWARE': '1', 'GALLIUM_DRIVER': 'llvmpipe',
        '__GLX_VENDOR_LIBRARY_NAME': 'mesa',
        '__EGL_VENDOR_LIBRARY_FILENAMES': '/usr/share/glvnd/egl_vendor.d/50_mesa.json',
        'LP_NUM_THREADS': '8', 'PYTHONUNBUFFERED': '1',
    }
    env_file = root/'services.env'
    env_file.write_text(''.join(f'{k}={json.dumps(v)}\n' for k,v in env.items()))
    env_file.chmod(0o644)
    unit_root = Path('/etc/systemd/system')
    names = []

    def unit(name, command, extra='', dependencies='', root_user=False):
        name = 'aerorozum-cloud-'+name+'.service'
        text = ('[Unit]\nDescription=AeroRozum cloud '+name+'\nWants=network-online.target\n'
                'After=network-online.target '+dependencies+'\nStartLimitIntervalSec=0\n'
                '[Service]\nType=simple\nUser='+('root' if root_user else 'aerorozum-app')+'\n'
                f'WorkingDirectory={release}\nEnvironmentFile={env_file}\n'
                'Restart=always\nRestartSec=5\nTimeoutStopSec=25\nKillMode=control-group\n'
                'NoNewPrivileges=true\nProtectHome=true\nPrivateTmp=true\nProtectSystem=strict\n'
                f'ReadWritePaths={root}\n'+extra+'ExecStart='+command+'\n'
                '[Install]\nWantedBy=multi-user.target\n')
        (unit_root/name).write_text(text)
        names.append(name)

    unit('gazebo', f'/usr/bin/xvfb-run -a -s "-screen 0 1280x800x24 -nolisten tcp" /usr/bin/gz sim -s -r -v 2 {sim}/worlds/aerorozum.sdf',
         'MemoryMax=8G\nCPUQuota=1000%\n')
    unit('adapter', f'/usr/bin/python3 {release}/simulation/runtime.py',
         'MemoryMax=3G\nCPUQuota=400%\n', 'aerorozum-cloud-gazebo.service')
    credential = Path('/etc/aerorozum/hackathon-key')
    if args.provider == 'gemini_api' and not credential.is_file():
        raise SystemExit('Install only the isolated hackathon key at '+str(credential)+' (mode 600)')
    extra = 'MemoryMax=5G\nCPUQuota=400%\n'
    if credential.is_file():
        extra += f'LoadCredential=gemini-api-key:{credential}\nEnvironment=AEROROZUM_GEMINI_KEY_FILE=%d/gemini-api-key\n'
    unit('api', f'{venv}/bin/python -m uvicorn aerorozum.api:app --host 127.0.0.1 --port 18901', extra,
         'aerorozum-cloud-adapter.service')
    unit('telemetry', f'{venv}/bin/python {release}/scripts/mirror_worker_observer.py',
         'MemoryMax=512M\nCPUQuota=100%\n', 'aerorozum-research-observer.service')
    adapter = '/opt/aerorozum/artifacts/training/real-transfer-machine-labels-v1/google_gemma-4-12B-it/real_sft/seed-17/adapter'
    unit('gemma', f'/opt/aerorozum/venv/bin/python -m uvicorn cloud.gemma_service:app --host 127.0.0.1 --port 18940',
         'MemoryMax=75G\nCPUQuota=500%\nEnvironment=CUDA_VISIBLE_DEVICES=0\n'
         'Environment=HF_HUB_OFFLINE=1\nEnvironment=TRANSFORMERS_OFFLINE=1\n'
         'Environment=AEROROZUM_GEMMA_WEIGHTS=/opt/aerorozum/models/gemma-4-12B-it\n'
         'Environment=AEROROZUM_GEMMA_ADAPTER='+adapter+'\n', root_user=True)
    # No public firewall rule: access via an authenticated SSH/IAP tunnel.
    Path('/etc/nginx/sites-available/aerorozum').write_text('''server {
    listen 127.0.0.1:18900;
    server_name _;
    client_max_body_size 1m;
    location / {
        proxy_pass http://127.0.0.1:18901;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_buffering off;
        proxy_read_timeout 240s;
    }
}
''')
    link = Path('/etc/nginx/sites-enabled/aerorozum')
    if not link.exists():
        link.symlink_to('/etc/nginx/sites-available/aerorozum')
    # The package's unused default site otherwise opens an unauthenticated port 80.
    Path('/etc/nginx/sites-enabled/default').unlink(missing_ok=True)
    run('nginx', '-t')
    run('systemctl', 'daemon-reload')
    run('systemctl', 'enable', *names, 'nginx.service')
    run('systemctl', 'restart', *names, 'nginx.service')
    # A host recovery can reset maxRunDuration. Keep a fixed guest deadline too,
    # and never extend it merely by reinstalling/restarting application services.
    marker = root/'runtime-deadline.json'
    if marker.exists():
        deadline = datetime.fromisoformat(json.loads(marker.read_text())['stop_at'])
    else:
        deadline = datetime.fromisoformat(args.stop_at.replace('Z','+00:00')) if args.stop_at else datetime.now(timezone.utc) + timedelta(hours=48)
        if deadline.tzinfo is None or deadline > datetime.now(timezone.utc)+timedelta(hours=48,minutes=5):
            raise ValueError('Deployment stop time must be timezone-aware and within 48 hours')
        marker.write_text(json.dumps({'stop_at': deadline.isoformat(), 'reason': 'User-authorized 48-hour deployment window'}))
    (unit_root/'aerorozum-cloud-deadline.service').write_text(
        '[Unit]\nDescription=AeroRozum fixed 48-hour runtime guard\n'
        '[Service]\nType=oneshot\nExecStart=/usr/bin/systemctl --no-block poweroff\n')
    (unit_root/'aerorozum-cloud-deadline.timer').write_text(
        '[Unit]\nDescription=AeroRozum fixed deployment deadline\n'
        '[Timer]\nOnCalendar='+deadline.strftime('%Y-%m-%d %H:%M:%S UTC')+'\n'
        'Persistent=true\nAccuracySec=1s\nUnit=aerorozum-cloud-deadline.service\n'
        '[Install]\nWantedBy=timers.target\n')
    run('systemctl', 'daemon-reload')
    run('systemctl', 'enable', '--now', 'aerorozum-cloud-deadline.timer')
    print(json.dumps({'units': names, 'provider': args.provider, 'access': 'SSH tunnel to loopback 18900',
                      'root': str(root), 'fixed_stop_at': deadline.isoformat(), 'research_validated': False}, indent=2))


if __name__ == '__main__':
    main()
