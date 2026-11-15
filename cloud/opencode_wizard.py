#!/usr/bin/env python3
"""Interactive, resumable installer for the published AeroRozum Harbor task."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

BASE = 'https://storage.googleapis.com/aerorozum-junction2026-submission-avid-relic-509408-e4'
ARCHIVE = 'aerorozum-harbor-close-inspection-v5.zip'
ARCHIVE_SHA = 'b49f5b89bd7fc64bbaff071da08bfaa145b65015daea0665f00708039c98de79'
HARBOR_VERSION = '0.26.0'
DEFAULT_MODEL = 'gemini-4-argon'
GIB = 1024 ** 3


class SetupError(Exception):
    pass


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def run(args, *, env=None, capture=False):
    try:
        result = subprocess.run([str(a) for a in args], env=env, check=True,
                                text=True, stdout=subprocess.PIPE if capture else None,
                                stderr=subprocess.PIPE if capture else None)
        return result.stdout if capture else None
    except subprocess.CalledProcessError as error:
        # Never echo a child environment or HTTP authentication headers.
        raise SetupError('Command failed (exit %s): %s' % (error.returncode, args[0])) from None


def ask(prompt, default=None):
    suffix = ' [%s]' % default if default is not None else ''
    try:
        # Buffered read/write streams require seeking, which a terminal does
        # not support. Separate handles also work when stdin carries a script.
        with open('/dev/tty', 'w') as output, open('/dev/tty', 'r') as terminal:
            output.write(prompt + suffix + ': ')
            output.flush()
            answer = terminal.readline()
    except OSError:
        raise SetupError('Run this wizard from an interactive terminal.') from None
    if not answer:
        raise SetupError('Terminal input closed; stopped without starting a run.')
    return answer.strip() or default


def confirm(prompt, default=True):
    return str(ask(prompt, 'yes' if default else 'no')).lower() in ('y', 'yes')


def download(url, destination, expected_sha=None):
    destination = Path(destination)
    if destination.exists() and expected_sha and sha(destination) == expected_sha:
        print('  Reusing verified download.')
        return
    partial = destination.with_suffix(destination.suffix + '.partial')
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as stream:
            total = int(response.headers.get('Content-Length', 0))
            count, last = 0, 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
                count += len(block)
                if time.monotonic() - last > 2:
                    print('\r  Downloaded %.0f / %.0f MB' % (count / 1e6, total / 1e6), end='', flush=True)
                    last = time.monotonic()
        print()
        if expected_sha and sha(partial) != expected_sha:
            raise SetupError('Download checksum differs. No downloaded code was executed.')
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(Path(root).resolve()):
        raise SetupError('Unsafe path in downloaded package.')
    return path


def extract_package(archive, directory):
    directory = Path(directory)
    package = directory / 'aerorozum-submission'
    if package.is_symlink():
        raise SetupError('The cached package directory is an unexpected symbolic link.')
    if not package.exists():
        staging = Path(tempfile.mkdtemp(prefix='.extract-', dir=directory))
        try:
            with zipfile.ZipFile(archive) as bundle:
                for entry in bundle.infolist():
                    safe_path(staging, entry.filename)
                    if stat.S_ISLNK(entry.external_attr >> 16):
                        raise SetupError('Package contains an unexpected symbolic link.')
                bundle.extractall(staging)
            (staging / 'aerorozum-submission').rename(package)
        finally:
            shutil.rmtree(staging)
    sums = package / 'SHA256SUMS'
    if not sums.is_file():
        raise SetupError('Package checksum manifest is missing.')
    with zipfile.ZipFile(archive) as original:
        if sums.read_bytes() != original.read('aerorozum-submission/SHA256SUMS'):
            raise SetupError('The cached package checksum manifest changed. Use a new installation directory.')
    checked = 0
    names = set()
    for line in sums.read_text().splitlines():
        expected, name = line.split('  ', 1)
        names.add(name)
        path = safe_path(package, name)
        if not path.is_file() or sha(path) != expected:
            raise SetupError('Package file changed: %s. Use a new installation directory.' % name)
        checked += 1
    for path in (package / 'thermal-investigation-budgeted').rglob('*'):
        if path.is_file() and path.relative_to(package).as_posix() not in names:
            raise SetupError('An extra file changed the cached task. Use a new installation directory.')
    print('  Verified archive and %s packaged files.' % checked)
    return package


def workspace(explicit=None):
    default = str(Path.home() / '.local/share/aerorozum-opencode')
    while True:
        directory = Path(explicit or ask('Where should the environment and results be saved?', default)).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(directory).free / GIB
        if free >= 20:
            return directory
        message = 'Only %.1f GiB free at %s; allow at least 20 GiB for setup and Docker builds.' % (free, directory)
        if explicit:
            raise SetupError(message)
        print(message + '\nChoose another writable drive, or press Ctrl+C and run this on a VM.')


def admin_prefix():
    if os.geteuid() == 0:
        return []
    if not shutil.which('sudo'):
        raise SetupError('Administrator access is needed for this setup step; sudo is unavailable.')
    run(['sudo', '-v'])
    return ['sudo']


def docker_ready(directory, env):
    docker = shutil.which('docker', path=env.get('PATH'))
    if not docker:
        if platform.system() != 'Linux':
            raise SetupError('Install and start Docker Desktop, then rerun the same launcher.')
        if not confirm('Docker is missing. Install Docker Engine and Compose using Docker’s installer?'):
            raise SetupError('Docker installation declined. Rerun after installing Docker + Compose.')
        installer = directory / 'docker-install.sh'
        download('https://get.docker.com', installer)
        run(admin_prefix() + ['sh', str(installer)])
        docker = shutil.which('docker')
        if not docker:
            raise SetupError('Docker was not found after installation.')
    def info(command):
        try:
            return json.loads(run(command + ['info', '--format', '{{json .}}'], capture=True))
        except (SetupError, ValueError):
            return None
    command = [docker]
    details = info(command)
    sudo_stop = threading.Event()
    if details is None:
        if platform.system() != 'Linux' or not confirm('Docker is unavailable. Check/start the local daemon using sudo?'):
            raise SetupError('Start Docker and grant this account access, then rerun.')
        prefix = admin_prefix()
        if shutil.which('systemctl'):
            run(prefix + ['systemctl', 'start', 'docker'])
        details = info(prefix + [docker])
        if details is None:
            raise SetupError('The local Docker daemon is still unavailable.')
        if prefix:
            # Avoid a logout/group change. Harbor uses this explicit sudo wrapper.
            bindir = directory / 'docker-access'
            bindir.mkdir(exist_ok=True)
            wrapper = bindir / 'docker'
            import shlex
            wrapper.write_text('#!/bin/sh\nexec sudo -n ' + shlex.quote(docker) + ' "$@"\n')
            wrapper.chmod(0o700)
            env['PATH'] = str(bindir) + os.pathsep + env.get('PATH', '')
            command = [str(wrapper)]
            def refresh():
                while not sudo_stop.wait(60):
                    subprocess.run(['sudo', '-n', '-v'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            threading.Thread(target=refresh, daemon=True).start()
    if details.get('OSType') != 'linux' or details.get('Architecture') not in ('x86_64', 'amd64'):
        raise SetupError('This prepared Gazebo task needs an amd64 Linux Docker engine. Use an x86_64 Linux VM.')
    try:
        run(command + ['compose', 'version'], env=env, capture=True)
    except SetupError:
        if platform.system() != 'Linux' or not confirm('Docker Compose is missing. Install the official Compose plugin?'):
            raise SetupError('Install Docker Compose v2 and rerun this launcher.')
        release = 'https://github.com/docker/compose/releases/download/v2.39.4/docker-compose-linux-x86_64'
        with urllib.request.urlopen(release + '.sha256', timeout=30) as response:
            checksum = response.read().decode().split()[0]
        if not re.fullmatch('[a-fA-F0-9]{64}', checksum):
            raise SetupError('Docker Compose checksum was malformed.')
        plugin = directory / 'docker-compose'
        download(release, plugin, checksum)
        prefix = admin_prefix()
        run(prefix + ['mkdir', '-p', '/usr/local/lib/docker/cli-plugins'])
        run(prefix + ['install', '-m', '755', str(plugin), '/usr/local/lib/docker/cli-plugins/docker-compose'])
        run(command + ['compose', 'version'], env=env, capture=True)
    print('  Docker ready: %s CPUs, %.1f GiB RAM.' % (details.get('NCPU', 0), details.get('MemTotal', 0) / GIB))
    if details.get('NCPU', 0) < 8 or details.get('MemTotal', 0) < 12 * GIB:
        if not confirm('Recommended: 8 CPU cores and 16 GB RAM. Continue with these smaller resources?', False):
            raise SetupError('Increase Docker resources or use a larger VM, then rerun.')
    graph = Path(details.get('DockerRootDir', '/var/lib/docker'))
    try:
        if graph.is_dir() and shutil.disk_usage(graph).free < 15 * GIB:
            raise SetupError('Docker storage has less than 15 GiB free. Free space or use a larger Docker disk.')
    except PermissionError:
        print('  Docker storage is administrator-owned; ensure at least 15 GiB is available there.')
    return sudo_stop


def harbor_ready(directory, env, existing=None):
    candidate = existing or shutil.which('harbor')
    if candidate:
        candidate = shutil.which(candidate) or candidate
        try:
            if HARBOR_VERSION in run([candidate, '--version'], capture=True):
                return str(Path(candidate).resolve())
        except SetupError:
            pass
    binary = directory / 'tools/uv'
    if not binary.exists():
        binary.parent.mkdir(parents=True, exist_ok=True)
        # Install the official binary wheel without requiring host pip/venv.
        # PyPI publishes the exact wheel digest; no shell installer is needed.
        with urllib.request.urlopen('https://pypi.org/pypi/uv/json', timeout=30) as response:
            metadata = json.load(response)
        machine = platform.machine().lower()
        architecture = 'aarch64' if machine in ('aarch64', 'arm64') else 'x86_64'
        family = 'macosx' if platform.system() == 'Darwin' else 'manylinux'
        if family == 'macosx' and architecture == 'aarch64':
            architecture = 'arm64'
        candidates = [item for item in metadata['urls'] if item['filename'].endswith('.whl')
                      and family in item['filename'] and architecture in item['filename']]
        if not candidates:
            raise SetupError('No UV binary wheel is available for this machine.')
        selected = sorted(candidates, key=lambda item: item['filename'])[0]
        wheel = directory / selected['filename']
        download(selected['url'], wheel, selected['digests']['sha256'])
        with zipfile.ZipFile(wheel) as bundle:
            entries = [n for n in bundle.namelist() if Path(n).name == 'uv']
            if len(entries) != 1:
                raise SetupError('The UV wheel did not contain exactly one UV executable.')
            binary.write_bytes(bundle.read(entries[0]))
        binary.chmod(0o700)
    venv = directory / 'harbor-venv'
    python = venv / 'bin/python'
    uv_env = dict(env, UV_PYTHON_INSTALL_DIR=str(directory / 'python'), UV_CACHE_DIR=str(directory / 'uv-cache'))
    if not python.is_file():
        run([binary, 'venv', '--python', '3.12', str(venv)], env=uv_env)
    run([binary, 'pip', 'install', '--python', python, 'harbor==' + HARBOR_VERSION], env=uv_env)
    return str(venv / 'bin/harbor')


def choose_model():
    existing = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_GENERATIVE_AI_API_KEY')
    while True:
        if existing and confirm('Use the Gemini API key already set in this terminal?'):
            key = existing
        else:
            key = getpass.getpass('Gemini API key (hidden; not saved): ').strip()
        existing = None
        if not key:
            print('An API key is required to run OpenCode with Gemini.')
            continue
        request = urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000',
                                         headers={'x-goog-api-key': key})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                document = json.load(response)
        except urllib.error.HTTPError as error:
            print('Gemini rejected the model-list request (HTTP %s). Check the API key and its restrictions.' % error.code)
            continue
        models = sorted({m['name'].removeprefix('models/') for m in document.get('models', [])
                         if 'generateContent' in m.get('supportedGenerationMethods', [])
                         and 'gemini' in m.get('name', '')
                         and not any(s in m['name'] for s in ('tts', 'embedding', 'image', 'audio', 'live'))})
        if not models:
            raise SetupError('This key exposes no Gemini generation models. Check the key’s project access.')
        default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
        limits = {m['name'].removeprefix('models/'): int(m.get('outputTokenLimit', 32768))
                  for m in document.get('models', [])}
        print('  Key accepted. Press Enter for the suggested model, or type ? to list models.')
        while True:
            model = ask('Gemini model', default)
            if model == '?':
                print('  ' + '\n  '.join(models))
            elif model not in models:
                print('That model is not available to this key. Type ? to list available IDs.')
            elif limits[model] < 512:
                print('That model’s output allowance is too small for this harness.')
            else:
                return key, model, min(32768, limits[model])


def runner_command(python, harbor, package, jobs, name, model, output_tokens=32768):
    return [str(python), str(package / 'run_harbor_trial.py'), '--task',
            str(package / 'thermal-investigation-budgeted'), '--job-name', name,
            '--jobs-dir', str(jobs), '--harbor', str(harbor), '--model', model,
            '--output-tokens', str(output_tokens)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--prepare-only', action='store_true', help='Set up and verify everything without asking for a key or running a model')
    parser.add_argument('--harbor', help='Reuse a Harbor 0.26.0 installation')
    args = parser.parse_args(argv)
    print('\nAeroRozum · OpenCode mission starter\n')
    directory = workspace(args.directory)
    env = os.environ.copy()
    stopped = docker_ready(directory, env)
    try:
        key, model, output_tokens = (None, None, None) if args.prepare_only else choose_model()
        print('\nPreparing the exact judging package…')
        archive = directory / ARCHIVE
        download(BASE + '/' + ARCHIVE, archive, ARCHIVE_SHA)
        package = extract_package(archive, directory)
        print('\nPreparing Harbor 0.26.0 and Python 3.12…')
        harbor = harbor_ready(directory, env, args.harbor)
        print('  Harbor ready. OpenCode 1.19.5 installs inside the task container automatically.')
        receipt = {'archive_sha256': ARCHIVE_SHA, 'harbor_version': HARBOR_VERSION,
                   'opencode_version': '1.19.5', 'package': str(package), 'contains_credentials': False}
        (directory / 'setup-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if args.prepare_only:
            print('\nSetup verified. Rerun the same launcher to enter your key and start OpenCode.')
            return 0
        name = 'opencode-' + time.strftime('%Y%m%d-%H%M%S')
        jobs = directory / 'jobs'
        print('\nReady: OpenCode + %s, native Gazebo, 75-second flight budget.' % model)
        print('First Docker build can take several minutes. Gemini requests use your key.')
        print('Results will be saved to ' + str(jobs / name))
        if not confirm('Start the mission now?'):
            print('Setup retained; no model run started.')
            return 0
        env['GEMINI_API_KEY'] = key
        env['GOOGLE_GENERATIVE_AI_API_KEY'] = key
        python = Path(harbor).parent / 'python'
        if not python.is_file():
            python = Path(sys.executable)
        command = runner_command(python, harbor, package, jobs, name, model, output_tokens)
        result = subprocess.run(command, env=env)
        rewards = sorted((jobs / name).glob('*/verifier/reward.txt'))
        print('\nRun artifacts: ' + str(jobs / name))
        for reward in rewards:
            print('Verifier score: ' + reward.read_text().strip())
        if not rewards:
            print('No verifier score was produced; inspect the job’s result.json and agent logs.')
        return result.returncode
    finally:
        stopped.set()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped. Downloads and setup are retained for the next launch.', file=sys.stderr)
        raise SystemExit(130)
    except EOFError:
        print('\nTerminal input closed; no further action started.', file=sys.stderr)
        raise SystemExit(1)
    except (SetupError, OSError, ValueError, zipfile.BadZipFile) as error:
        print('\nSetup stopped: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
