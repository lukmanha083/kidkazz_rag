"""Tests for kidkazz db sync command."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli.commands.db import (
    _check_fly_cli,
    _get_data_size,
    _get_fly_app,
    _get_local_data_dir,
    app,
)

runner = CliRunner()


class TestCheckFlyCli:
    """Tests for _check_fly_cli helper."""

    @patch("src.cli.commands.db.subprocess.run")
    def test_returns_version_when_installed(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="fly v0.3.50 linux/amd64"
        )
        result = _check_fly_cli()
        assert "fly" in result

    @patch("src.cli.commands.db.subprocess.run", side_effect=FileNotFoundError)
    def test_raises_when_not_installed(self, mock_run):
        with pytest.raises(FileNotFoundError, match="fly CLI not found"):
            _check_fly_cli()


class TestGetFlyApp:
    """Tests for _get_fly_app helper."""

    def test_returns_flag_when_provided(self):
        assert _get_fly_app("my-app") == "my-app"

    def test_reads_fly_toml(self, tmp_path, monkeypatch):
        fly_toml = tmp_path / "fly.toml"
        fly_toml.write_text('app = "kidkazz-rag"\n')
        monkeypatch.chdir(tmp_path)
        assert _get_fly_app(None) == "kidkazz-rag"

    def test_flag_overrides_toml(self, tmp_path, monkeypatch):
        fly_toml = tmp_path / "fly.toml"
        fly_toml.write_text('app = "kidkazz-rag"\n')
        monkeypatch.chdir(tmp_path)
        assert _get_fly_app("other-app") == "other-app"

    def test_raises_when_no_toml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="Could not determine Fly app name"):
            _get_fly_app(None)


class TestGetDataSize:
    """Tests for _get_data_size helper."""

    def test_bytes(self, tmp_path):
        (tmp_path / "small.txt").write_bytes(b"x" * 500)
        assert "500 B" == _get_data_size(tmp_path)

    def test_kilobytes(self, tmp_path):
        (tmp_path / "file.bin").write_bytes(b"x" * 2048)
        assert "KB" in _get_data_size(tmp_path)

    def test_megabytes(self, tmp_path):
        (tmp_path / "file.bin").write_bytes(b"x" * (2 * 1024 * 1024))
        assert "MB" in _get_data_size(tmp_path)


class TestGetLocalDataDir:
    """Tests for _get_local_data_dir helper."""

    @patch("src.cli.commands.db._find_helix_dir")
    def test_raises_when_no_data(self, mock_helix):
        mock_helix.return_value = Path("/fake/.helix/dev")
        with pytest.raises(FileNotFoundError, match="Local Helix data not found"):
            _get_local_data_dir()

    @patch("src.cli.commands.db._find_helix_dir")
    def test_raises_when_no_mdb(self, mock_helix, tmp_path):
        # Create directory structure but no data.mdb
        helix_dev = tmp_path / ".helix" / "dev"
        user_dir = tmp_path / ".helix" / ".volumes" / "dev" / "user"
        user_dir.mkdir(parents=True)
        mock_helix.return_value = helix_dev
        with pytest.raises(FileNotFoundError, match="No data.mdb"):
            _get_local_data_dir()

    @patch("src.cli.commands.db._find_helix_dir")
    def test_returns_path_when_valid(self, mock_helix, tmp_path):
        helix_dev = tmp_path / ".helix" / "dev"
        helix_dev.mkdir(parents=True)
        user_dir = tmp_path / ".helix" / ".volumes" / "dev" / "user"
        user_dir.mkdir(parents=True)
        (user_dir / "data.mdb").write_bytes(b"fake mdb")
        mock_helix.return_value = helix_dev
        assert _get_local_data_dir() == user_dir


class TestSyncCommand:
    """Tests for the sync command via CLI runner."""

    @patch("src.cli.commands.db._get_fly_app")
    @patch("src.cli.commands.db._check_fly_cli")
    @patch("src.cli.commands.db._get_local_data_dir")
    def test_dry_run(self, mock_data, mock_cli, mock_app):
        mock_data.return_value = Path("/fake/data/user")
        mock_cli.return_value = "fly v0.3.50"
        mock_app.return_value = "kidkazz-rag"

        with patch.object(Path, "rglob", return_value=[]):
            result = runner.invoke(app, ["sync", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry run" in result.output or "dry_run" in result.output

    @patch("src.cli.commands.db._check_fly_cli", side_effect=FileNotFoundError("fly CLI not found"))
    @patch("src.cli.commands.db._get_local_data_dir")
    def test_no_fly_cli(self, mock_data, mock_cli):
        mock_data.return_value = Path("/fake/data/user")

        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 1
        assert "fly CLI not found" in result.output

    @patch("src.cli.commands.db._get_local_data_dir", side_effect=FileNotFoundError("Local Helix data not found"))
    def test_no_local_data(self, mock_data):
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 1
        assert "Local Helix data not found" in result.output

    @patch("src.cli.commands.db._get_fly_app")
    @patch("src.cli.commands.db._check_fly_cli")
    @patch("src.cli.commands.db._get_local_data_dir")
    def test_dry_run_json(self, mock_data, mock_cli, mock_app):
        mock_data.return_value = Path("/fake/data/user")
        mock_cli.return_value = "fly v0.3.50"
        mock_app.return_value = "kidkazz-rag"

        with patch.object(Path, "rglob", return_value=[]):
            result = runner.invoke(app, ["sync", "--dry-run", "--json"])

        assert result.exit_code == 0
        assert "dry_run" in result.output
        assert "kidkazz-rag" in result.output

    @patch("src.cli.commands.db._get_fly_app")
    @patch("src.cli.commands.db._check_fly_cli")
    @patch("src.cli.commands.db._get_local_data_dir")
    def test_app_flag_overrides(self, mock_data, mock_cli, mock_app):
        mock_data.return_value = Path("/fake/data/user")
        mock_cli.return_value = "fly v0.3.50"
        mock_app.return_value = "my-custom-app"

        with patch.object(Path, "rglob", return_value=[]):
            result = runner.invoke(app, ["sync", "--dry-run", "--app", "my-custom-app"])

        assert result.exit_code == 0
        # Verify the --app flag was passed through
        mock_app.assert_called_once_with("my-custom-app")
