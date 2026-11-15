import hashlib
import importlib.util
import io
import json
import errno
import os
from pathlib import Path
import pty
import select
import signal
import stat
import subprocess
import sys
import time
import zipfile

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'cloud' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wizard = load('opencode_wizard')


@pytest.mark.parametrize('answer,expected', [(b'chosen\n', 'chosen'), (b'\n', 'default')])
def test_prompt_uses_real_controlling_terminal_even_with_piped_stdin(answer, expected):
    source = str(Path(wizard.__file__).resolve())
    program = ("import importlib.util,io,sys; "
               f"s=importlib.util.spec_from_file_location('wizard',{source!r}); "
               "w=importlib.util.module_from_spec(s);s.loader.exec_module(w); "
               "sys.stdin=io.StringIO('must not read this\\n'); "
               "print('ANSWER='+w.ask('Terminal test','default'),flush=True)")
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, '-c', program])
    output, answered, reaped = b'', False, False
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if not select.select([fd], [], [], .1)[0]:
                continue
            try:
                block = os.read(fd, 65536)
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
            if not block:
                break
            output += block
            if b'Terminal test [default]: ' in output and not answered:
                os.write(fd, answer)
                answered = True
        assert answered, output.decode(errors='replace')
        assert ('ANSWER=' + expected).encode() in output, output.decode(errors='replace')
        _, status = os.waitpid(pid, 0)
        reaped = True
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        os.close(fd)
        if not reaped:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)


def test_prompt_rejects_a_session_without_any_controlling_terminal():
    source = str(Path(wizard.__file__).resolve())
    program = ("import importlib.util; "
               f"s=importlib.util.spec_from_file_location('wizard',{source!r}); "
               "w=importlib.util.module_from_spec(s);s.loader.exec_module(w);w.ask('Prompt')")
    result = subprocess.run([sys.executable, '-c', program], start_new_session=True,
                            input='not a terminal\n', capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'Run this wizard from an interactive terminal.' in result.stderr


def package(tmp_path, extra=None):
    path = tmp_path / 'fixture.zip'
    content = b'example scene bytes'
    relative = 'thermal-investigation-budgeted/environment/example.txt'
    sums = hashlib.sha256(content).hexdigest() + '  ' + relative + '\n'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('aerorozum-submission/' + relative, content)
        archive.writestr('aerorozum-submission/SHA256SUMS', sums)
        if extra:
            extra(archive)
    return path


def test_package_can_resume_but_rejects_modified_task_or_checksum_list(tmp_path):
    archive = package(tmp_path)
    root = wizard.extract_package(archive, tmp_path)
    assert wizard.extract_package(archive, tmp_path) == root
    target = root / 'thermal-investigation-budgeted/environment/example.txt'
    target.write_text('different scene')
    with pytest.raises(wizard.SetupError, match='Package file changed'):
        wizard.extract_package(archive, tmp_path)
    (root / 'SHA256SUMS').write_text(hashlib.sha256(target.read_bytes()).hexdigest() + '  thermal-investigation-budgeted/environment/example.txt\n')
    with pytest.raises(wizard.SetupError, match='manifest changed'):
        wizard.extract_package(archive, tmp_path)


def test_archive_cannot_escape_or_introduce_symlinks(tmp_path):
    def symlink(archive):
        entry = zipfile.ZipInfo('aerorozum-submission/link')
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(entry, '/tmp/private')
    with pytest.raises(wizard.SetupError, match='symbolic link'):
        wizard.extract_package(package(tmp_path, symlink), tmp_path)
    with pytest.raises(wizard.SetupError, match='Unsafe path'):
        wizard.extract_package(package(tmp_path, lambda z: z.writestr('../escaped', 'bad')), tmp_path)
    assert not (tmp_path.parent / 'escaped').exists()


def test_extra_task_file_is_not_silently_used_in_docker_build(tmp_path):
    archive = package(tmp_path)
    root = wizard.extract_package(archive, tmp_path)
    (root / 'thermal-investigation-budgeted/environment/extra.py').write_text('changed runtime')
    with pytest.raises(wizard.SetupError, match='extra file'):
        wizard.extract_package(archive, tmp_path)


def test_runner_uses_pinned_opencode_wrapper_without_credentials_in_arguments(tmp_path):
    command = wizard.runner_command('/venv/python', '/venv/harbor', tmp_path, tmp_path / 'jobs', 'trial', 'gemini-4-argon')
    assert command[1] == str(tmp_path / 'run_harbor_trial.py')
    assert command[command.index('--output-tokens') + 1] == '32768'
    assert '--key-file' not in command and '--api-key' not in command


def test_bootstrap_is_bound_to_exact_python_source():
    builder = load('build_opencode_launcher')
    source = Path(wizard.__file__)
    rendered = builder.render(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() in rendered
    assert 'read -r -p' in rendered and '</dev/tty' in rendered
    assert 'WIZARD_CHECKSUM' not in rendered


def test_enter_respects_negative_default_for_resource_override(monkeypatch):
    monkeypatch.setattr(wizard, 'ask', lambda prompt, default: default)
    assert wizard.confirm('Install?', True) is True
    assert wizard.confirm('Continue with insufficient resources?', False) is False


def test_key_stays_in_header_and_selected_model_output_limit_is_respected(monkeypatch, capsys):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GOOGLE_GENERATIVE_AI_API_KEY', raising=False)
    monkeypatch.setattr(wizard.getpass, 'getpass', lambda _: 'private-test-key')
    monkeypatch.setattr(wizard, 'ask', lambda *args: 'gemini-small')
    def respond(request, timeout):
        assert 'private-test-key' not in request.full_url
        assert request.get_header('X-goog-api-key') == 'private-test-key'
        return io.BytesIO(json.dumps({'models': [{'name': 'models/gemini-small',
            'supportedGenerationMethods': ['generateContent'], 'outputTokenLimit': 8192}]}).encode())
    monkeypatch.setattr(wizard.urllib.request, 'urlopen', respond)
    key, model, tokens = wizard.choose_model()
    assert (model, tokens) == ('gemini-small', 8192)
    assert key == 'private-test-key' and key not in capsys.readouterr().out
