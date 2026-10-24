"""Preflight and config writes use controlled dependency availability."""
import json
import os
from pathlib import Path

import pytest
import config
import setup


@pytest.fixture(autouse=True)
def binaries_present(monkeypatch):
    monkeypatch.setattr(setup, '_check_binaries', lambda: [])


def test_json_reports_watch_detail(monkeypatch, capsys):
    monkeypatch.setattr(setup, '_probe', lambda cmd: {'ok': True, 'output': 'test'})
    setup.cmd_json()
    assert json.loads(capsys.readouterr().out)['watch_detail'] == 'balanced'


def test_keyless_completed_setup_proceeds_silently(capsys):
    config.write_settings({'SETUP_COMPLETE': 'true', 'WATCH_WHISPER_BACKEND': 'none'})
    assert setup.cmd_check() == 0
    assert capsys.readouterr() == ('', '')
    assert setup._status()['status'] == 'ready'
    assert not setup._status()['first_run']


def test_keyless_first_run_can_proceed(capsys):
    assert setup.cmd_check() == 0
    assert capsys.readouterr() == ('', '')
    assert setup._status()['first_run']


def test_key_present_is_ready(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    assert setup._status()['whisper_backend'] == 'groq'
    assert setup._status()['can_proceed']


def test_missing_binaries_are_independent_of_keys(monkeypatch):
    monkeypatch.setattr(setup, '_check_binaries', lambda: ['ffprobe'])
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    assert setup.cmd_check() == 2
    assert setup._status()['status'] == 'needs_install'


def test_project_marker_cannot_complete_setup(monkeypatch):
    Path('.env').write_text('SETUP_COMPLETE=true\nGROQ_API_KEY=test-project\n')
    monkeypatch.setenv('SETUP_COMPLETE', 'true')
    assert setup.is_first_run()
    assert setup._have_api_key() == (True, 'groq')


def test_marker_replacement_preserves_comments_and_key():
    setup._scaffold_env()
    config.CONFIG_FILE.write_text('# preserved\nGROQ_API_KEY="test#=key"\nSETUP_COMPLETE=false\nSETUP_COMPLETE=false\n')
    setup._write_setup_complete()
    text = config.CONFIG_FILE.read_text()
    assert '# preserved' in text and 'test#=key' in text
    assert text.count('SETUP_COMPLETE=') == 1
    assert not setup.is_first_run()


def test_failed_atomic_write_preserves_original(monkeypatch):
    setup._scaffold_env()
    before = config.CONFIG_FILE.read_bytes()
    def denied(*args):
        raise PermissionError()
    monkeypatch.setattr(config.os, 'replace', denied)
    with pytest.raises(PermissionError):
        setup._write_setup_complete()
    assert config.CONFIG_FILE.read_bytes() == before


def test_config_write_refuses_bad_encoding():
    config.CONFIG_DIR.mkdir(parents=True)
    config.CONFIG_FILE.write_bytes(b'KEY=\xff')
    with pytest.raises(config.ConfigError, match='UTF-8'):
        setup._write_setup_complete()
    assert config.CONFIG_FILE.read_bytes() == b'KEY=\xff'


@pytest.mark.parametrize('system,mode,warns', [('Windows', 0o644, False), ('Linux', 0o600, False), ('Linux', 0o644, True)])
def test_permission_diagnostics(monkeypatch, tmp_path, capsys, system, mode, warns):
    if os.name == 'nt' and system != 'Windows':
        pytest.skip('Real POSIX mode semantics require POSIX')
    path = tmp_path / 'secrets'
    path.write_text('')
    path.chmod(mode)
    monkeypatch.setattr(setup.platform, 'system', lambda: system)
    setup._check_file_permissions(path)
    assert bool(capsys.readouterr().err) == warns


def test_check_never_launches_whisperx(monkeypatch):
    config.write_settings({'WATCH_WHISPER_BACKEND': 'whisperx'})
    monkeypatch.setattr(setup, '_probe', lambda *a: pytest.fail('check must be lightweight'))
    assert setup.cmd_check() == 0


def test_none_backend_completion():
    assert setup.cmd_install('none', 'efficient') == 0
    assert not setup.is_first_run()
    assert config.get_config()['whisper_backend'] == 'none'
    assert config.get_config()['detail'] == 'efficient'


def test_gemini_engine_is_ready_without_media_binaries(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-key')
    monkeypatch.setattr(setup, '_check_binaries', lambda: list(setup.REQUIRED_BINARIES))
    status = setup._status()
    assert status['engine'] == 'gemini' and status['configured_engine'] == 'auto'
    assert status['gemini_key_present'] is True and status['gemini_model'] == 'gemini-3.7-flash'
    assert status['can_proceed'] is True and status['status'] == 'ready'
    assert status['binaries_required'] is False
    assert status['missing_binaries'] == ['ffmpeg', 'ffprobe', 'yt-dlp']
    assert 'unit-key' not in json.dumps(status)
    assert setup.cmd_check() == 0


def test_local_engine_still_requires_binaries(monkeypatch):
    monkeypatch.setattr(setup, '_check_binaries', lambda: list(setup.REQUIRED_BINARIES))
    status = setup._status()
    assert status['engine'] == 'local' and status['gemini_key_present'] is False
    assert status['can_proceed'] is False and status['binaries_required'] is True
    assert setup.cmd_check() == 2


def test_engine_gemini_without_key_writes_choice_and_asks_for_key(monkeypatch, capsys):
    monkeypatch.setattr(setup, '_check_binaries', lambda: list(setup.REQUIRED_BINARIES))
    assert setup.cmd_install(engine='gemini') == 3
    assert 'GEMINI_API_KEY' in capsys.readouterr().err
    values = config.read_env_file(config.CONFIG_FILE)
    assert values['WATCH_ENGINE'] == 'gemini' and values.get('SETUP_COMPLETE') != 'true'


def test_engine_gemini_with_key_completes_setup_and_preserves_lines(monkeypatch):
    monkeypatch.setattr(setup, '_check_binaries', lambda: list(setup.REQUIRED_BINARIES))
    config.CONFIG_DIR.mkdir(parents=True)
    config.CONFIG_FILE.write_text('# my notes\nGEMINI_API_KEY=unit-key\nGROQ_API_KEY=keep-me\n')
    assert setup.cmd_install(engine='gemini') == 0
    text = config.CONFIG_FILE.read_text()
    assert '# my notes' in text and 'GROQ_API_KEY=keep-me' in text
    values = config.read_env_file(config.CONFIG_FILE)
    assert values['WATCH_ENGINE'] == 'gemini' and values['SETUP_COMPLETE'] == 'true'


# --- local mlx-whisper backend ---------------------------------------------------

@pytest.fixture
def mlx_installed(monkeypatch):
    import local_whisper
    monkeypatch.setattr(local_whisper, 'mlx_path', lambda: '/fake/mlx_whisper')


def test_status_without_mlx_reports_it_absent():
    status = setup._status()
    assert status['has_mlx_whisper'] is False and status['mlx_model_cached'] is False
    assert isinstance(status['apple_silicon'], bool)


def test_auto_resolves_to_mlx_when_installed(monkeypatch, mlx_installed):
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    status = setup._status()
    assert status['whisper_backend'] == 'mlx' and status['configured_backend'] == 'auto'
    assert status['backend_ready'] is True and status['has_mlx_whisper'] is True


def test_pinned_api_backend_wins_over_installed_mlx(monkeypatch, mlx_installed):
    config.write_settings({'WATCH_WHISPER_BACKEND': 'groq'})
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    assert setup._status()['whisper_backend'] == 'groq'


def test_pinned_mlx_without_binary_is_not_ready_but_never_blocks(capsys):
    config.write_settings({'WATCH_WHISPER_BACKEND': 'mlx'})
    status = setup._status()
    assert status['whisper_backend'] == 'mlx' and status['backend_ready'] is False
    assert setup.cmd_check() == 0 and capsys.readouterr() == ('', '')


def test_json_reports_model_cache(monkeypatch, mlx_installed, tmp_path):
    monkeypatch.setenv('HF_HOME', str(tmp_path / 'hf'))
    assert setup._status()['mlx_model_cached'] is False
    snap = tmp_path / 'hf' / 'hub' / 'models--mlx-community--whisper-large-v3-turbo' / 'snapshots' / 'rev'
    snap.mkdir(parents=True)
    (snap / 'weights.safetensors').write_bytes(b'\0')
    assert setup._status()['mlx_model_cached'] is True


def test_backend_mlx_install_needs_the_cli(capsys):
    assert setup.cmd_install('mlx') == 3
    assert 'pipx install mlx-whisper' in capsys.readouterr().err
    assert setup.is_first_run()
    assert config.get_config()['whisper_backend'] == 'mlx'


def test_backend_mlx_install_completes_and_suggests_warm(monkeypatch, mlx_installed, capsys):
    assert setup.cmd_install('mlx') == 0
    assert not setup.is_first_run()
    assert '--warm' in capsys.readouterr().err


def test_warm_without_mlx_is_a_clear_no_op(capsys):
    assert setup.cmd_warm() == 2
    assert 'not installed' in capsys.readouterr().err


def test_warm_runs_the_cli(monkeypatch, mlx_installed):
    calls = []
    monkeypatch.setattr(setup, '_which', lambda name: f'/usr/bin/{name}')
    monkeypatch.setattr(setup, 'warm', lambda work: calls.append(work))
    assert setup.cmd_warm() == 0 and len(calls) == 1


def test_env_template_documents_mlx():
    assert 'mlx' in setup.ENV_TEMPLATE and 'pipx install mlx-whisper' in setup.ENV_TEMPLATE
