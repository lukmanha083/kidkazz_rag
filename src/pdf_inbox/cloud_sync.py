"""Cloud sync functionality using rclone.

This module provides the CloudSync class that wraps rclone subprocess calls
for syncing PDF files between local inbox and cloud storage providers.
"""

import subprocess
from pathlib import Path
from typing import Optional

from src.pdf_inbox.models import SyncResult


class CloudSync:
    """Wrapper for rclone cloud synchronization.

    This class provides methods for:
    - Checking rclone installation
    - Listing configured remotes
    - Uploading local files to cloud storage
    - Downloading files from cloud storage
    - Listing remote files

    Attributes:
        remote_name: Name of the rclone remote (e.g., "gdrive").
        remote_path: Path on the remote (e.g., "kidkazz_inbox").
    """

    def __init__(
        self,
        remote_name: str = "",
        remote_path: str = "",
    ) -> None:
        """Initialize CloudSync.

        Args:
            remote_name: Name of the rclone remote.
            remote_path: Path on the remote storage.
        """
        self.remote_name = remote_name
        self.remote_path = remote_path

    def check_rclone_installed(self) -> bool:
        """Check if rclone is installed.

        Returns:
            True if rclone is installed and accessible, False otherwise.
        """
        try:
            result = subprocess.run(
                ["rclone", "version"],
                capture_output=True,
                timeout=30,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False

    def get_rclone_version(self) -> str:
        """Get the installed rclone version.

        Returns:
            Version string or empty string if not installed.
        """
        try:
            result = subprocess.run(
                ["rclone", "version"],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                output = result.stdout.decode().strip()
                # Extract version from first line
                first_line = output.split("\n")[0]
                return first_line.replace("rclone ", "")
            return ""
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""

    def list_remotes(self) -> list[str]:
        """List configured rclone remotes.

        Returns:
            List of remote names (without trailing colon).
        """
        try:
            result = subprocess.run(
                ["rclone", "listremotes"],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                output = result.stdout.decode().strip()
                if not output:
                    return []
                # Each remote ends with ":", remove it
                return [r.rstrip(":") for r in output.split("\n") if r]
            return []
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    def validate_remote(self, remote_name: str) -> bool:
        """Check if a remote is configured.

        Args:
            remote_name: Name of the remote to validate.

        Returns:
            True if the remote is configured, False otherwise.
        """
        remotes = self.list_remotes()
        return remote_name in remotes

    def sync_to_remote(
        self,
        local_path: Path,
        dry_run: bool = False,
    ) -> SyncResult:
        """Upload local files to cloud storage.

        Args:
            local_path: Local directory to sync from.
            dry_run: If True, show what would be synced without syncing.

        Returns:
            SyncResult with success status and details.
        """
        # Check if remote is configured
        if not self.remote_name:
            return SyncResult(
                success=False,
                files_synced=0,
                bytes_transferred=0,
                direction="upload",
                error_message="No remote configured. Set remote_name first.",
            )

        # Build remote destination
        remote_dest = f"{self.remote_name}:{self.remote_path}"

        # Build command
        cmd = ["rclone", "copy", str(local_path), remote_dest, "--verbose"]
        if dry_run:
            cmd.append("--dry-run")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=3600,  # 1 hour timeout for large syncs
            )

            if result.returncode == 0:
                # rclone writes verbose/progress output to stderr
                output = result.stderr.decode()
                return SyncResult(
                    success=True,
                    files_synced=self._parse_files_count(output),
                    bytes_transferred=self._parse_bytes_transferred(output),
                    direction="upload",
                )
            else:
                error_msg = result.stderr.decode().strip()
                return SyncResult(
                    success=False,
                    files_synced=0,
                    bytes_transferred=0,
                    direction="upload",
                    error_message=error_msg or "Sync failed",
                )

        except subprocess.TimeoutExpired:
            return SyncResult(
                success=False,
                files_synced=0,
                bytes_transferred=0,
                direction="upload",
                error_message="Sync timeout after 1 hour",
            )
        except FileNotFoundError:
            return SyncResult(
                success=False,
                files_synced=0,
                bytes_transferred=0,
                direction="upload",
                error_message="rclone not found. Please install rclone.",
            )

    def sync_from_remote(
        self,
        local_path: Path,
        dry_run: bool = False,
    ) -> SyncResult:
        """Download files from cloud storage to local.

        Args:
            local_path: Local directory to sync to.
            dry_run: If True, show what would be synced without syncing.

        Returns:
            SyncResult with success status and details.
        """
        # Check if remote is configured
        if not self.remote_name:
            return SyncResult(
                success=False,
                files_synced=0,
                bytes_transferred=0,
                direction="download",
                error_message="No remote configured. Set remote_name first.",
            )

        # Build remote source
        remote_src = f"{self.remote_name}:{self.remote_path}"

        # Build command
        cmd = ["rclone", "copy", remote_src, str(local_path), "--verbose"]
        if dry_run:
            cmd.append("--dry-run")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=3600,  # 1 hour timeout for large syncs
            )

            if result.returncode == 0:
                # rclone writes verbose/progress output to stderr
                output = result.stderr.decode()
                return SyncResult(
                    success=True,
                    files_synced=self._parse_files_count(output),
                    bytes_transferred=self._parse_bytes_transferred(output),
                    direction="download",
                )
            else:
                error_msg = result.stderr.decode().strip()
                return SyncResult(
                    success=False,
                    files_synced=0,
                    bytes_transferred=0,
                    direction="download",
                    error_message=error_msg or "Sync failed",
                )

        except subprocess.TimeoutExpired:
            return SyncResult(
                success=False,
                files_synced=0,
                bytes_transferred=0,
                direction="download",
                error_message="Sync timeout after 1 hour",
            )
        except FileNotFoundError:
            return SyncResult(
                success=False,
                files_synced=0,
                bytes_transferred=0,
                direction="download",
                error_message="rclone not found. Please install rclone.",
            )

    def list_remote_files(self) -> list[dict[str, any]]:
        """List files on the remote.

        Returns:
            List of dicts with 'name' and 'size' keys.
        """
        if not self.remote_name:
            return []

        remote_path = f"{self.remote_name}:{self.remote_path}"

        try:
            result = subprocess.run(
                ["rclone", "ls", remote_path],
                capture_output=True,
                timeout=60,
            )

            if result.returncode == 0:
                return self._parse_ls_output(result.stdout.decode())
            return []

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    def _parse_ls_output(self, output: str) -> list[dict[str, any]]:
        """Parse rclone ls output.

        Args:
            output: Output from 'rclone ls' command.

        Returns:
            List of dicts with 'name' and 'size' keys.
        """
        files = []
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            # Format: "  size filename"
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                try:
                    size = int(parts[0])
                    name = parts[1]
                    files.append({"size": size, "name": name})
                except ValueError:
                    continue
        return files

    def _parse_files_count(self, output: str) -> int:
        """Parse file count from rclone output.

        Args:
            output: Output from rclone command.

        Returns:
            Number of files transferred or 0.
        """
        # Look for pattern like "Transferred: 5 files"
        for line in output.split("\n"):
            if "files" in line.lower() and "transferred" in line.lower():
                parts = line.split()
                for part in parts:
                    try:
                        return int(part)
                    except ValueError:
                        continue
        return 0

    def _parse_bytes_transferred(self, output: str) -> int:
        """Parse bytes transferred from rclone output.

        Args:
            output: Output from rclone command.

        Returns:
            Bytes transferred or 0.
        """
        # Simplified parsing - would need more sophisticated parsing
        # for production use
        return 0
