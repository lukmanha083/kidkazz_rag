"""Cloud sync functionality using rclone.

This module provides the CloudSync class that wraps rclone subprocess calls
for syncing PDF files between local inbox and cloud storage providers.
"""

import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from src.pdf_inbox.models import SyncResult


class SyncProgress:
    """Progress information for a sync operation."""

    def __init__(
        self,
        current_file: str = "",
        files_completed: int = 0,
        total_files: int = 0,
        bytes_transferred: int = 0,
        total_bytes: int = 0,
    ):
        self.current_file = current_file
        self.files_completed = files_completed
        self.total_files = total_files
        self.bytes_transferred = bytes_transferred
        self.total_bytes = total_bytes


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

    def count_pending_uploads(self, local_path: Path) -> list[dict]:
        """Count files that need to be uploaded.

        Uses rclone check to find files missing on remote.

        Args:
            local_path: Local directory to check.

        Returns:
            List of dicts with 'name' and 'size' for files needing upload.
        """
        if not self.remote_name:
            return []

        remote_dest = f"{self.remote_name}:{self.remote_path}"

        try:
            # Use rclone check --one-way to find files missing on remote
            result = subprocess.run(
                [
                    "rclone", "check", str(local_path), remote_dest,
                    "--one-way", "--missing-on-dst", "/dev/stderr",
                ],
                capture_output=True,
                timeout=120,
            )
            # Parse missing files from stderr
            # Format: "ERROR : file.pdf: File not in Dest"
            files = []
            for line in result.stderr.decode().split("\n"):
                if "File not in" in line or "Sizes differ" in line:
                    # Extract filename between first and second colon
                    match = re.search(r":\s*(.+?):\s*(?:File not in|Sizes differ)", line)
                    if match:
                        filename = match.group(1).strip()
                        filepath = local_path / filename
                        size = filepath.stat().st_size if filepath.exists() else 0
                        files.append({"name": filename, "size": size})
            return files
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    def count_pending_downloads(self, local_path: Path) -> list[dict]:
        """Count files that need to be downloaded.

        Uses rclone check to find files missing locally.

        Args:
            local_path: Local directory to check against.

        Returns:
            List of dicts with 'name' and 'size' for files needing download.
        """
        if not self.remote_name:
            return []

        remote_src = f"{self.remote_name}:{self.remote_path}"

        try:
            # Use rclone check --one-way (reversed) to find files missing locally
            result = subprocess.run(
                [
                    "rclone", "check", remote_src, str(local_path),
                    "--one-way", "--missing-on-dst", "/dev/stderr",
                ],
                capture_output=True,
                timeout=120,
            )
            # Parse missing files from stderr
            files = []
            for line in result.stderr.decode().split("\n"):
                if "File not in" in line or "Sizes differ" in line:
                    match = re.search(r":\s*(.+?):\s*(?:File not in|Sizes differ)", line)
                    if match:
                        filename = match.group(1).strip()
                        files.append({"name": filename, "size": 0})
            return files
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    def sync_to_remote_with_progress(
        self,
        local_path: Path,
        progress_callback: Callable[[SyncProgress], None],
        dry_run: bool = False,
    ) -> SyncResult:
        """Upload local files to cloud storage with progress updates.

        Args:
            local_path: Local directory to sync from.
            progress_callback: Called with SyncProgress for each file.
            dry_run: If True, show what would be synced without syncing.

        Returns:
            SyncResult with success status and details.
        """
        if not self.remote_name:
            return SyncResult(
                success=False,
                files_synced=0,
                bytes_transferred=0,
                direction="upload",
                error_message="No remote configured. Set remote_name first.",
            )

        remote_dest = f"{self.remote_name}:{self.remote_path}"

        # Build command with verbose output for progress parsing
        cmd = ["rclone", "copy", str(local_path), remote_dest, "--verbose"]
        if dry_run:
            cmd.append("--dry-run")

        try:
            # Use Popen to stream output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            files_completed = 0
            output_lines = []

            # Read stderr line by line (rclone outputs to stderr with --verbose)
            for line in process.stderr:
                output_lines.append(line)

                # Parse progress from verbose output
                if ": Copied" in line or "Would copy" in line:
                    files_completed += 1
                    # Extract filename
                    match = re.search(r":\s*(.+?):\s*(?:Copied|Would copy)", line)
                    current_file = match.group(1) if match else ""
                    progress_callback(SyncProgress(
                        current_file=current_file,
                        files_completed=files_completed,
                    ))

            process.wait()

            if process.returncode == 0:
                output = "".join(output_lines)
                return SyncResult(
                    success=True,
                    files_synced=files_completed,
                    bytes_transferred=self._parse_bytes_transferred(output),
                    direction="upload",
                )
            else:
                error_msg = "".join(output_lines).strip()
                return SyncResult(
                    success=False,
                    files_synced=0,
                    bytes_transferred=0,
                    direction="upload",
                    error_message=error_msg or "Sync failed",
                )

        except FileNotFoundError:
            return SyncResult(
                success=False,
                files_synced=0,
                bytes_transferred=0,
                direction="upload",
                error_message="rclone not found. Please install rclone.",
            )

    def sync_from_remote_with_progress(
        self,
        local_path: Path,
        progress_callback: Callable[[SyncProgress], None],
        dry_run: bool = False,
    ) -> SyncResult:
        """Download files from cloud storage with progress updates.

        Args:
            local_path: Local directory to sync to.
            progress_callback: Called with SyncProgress for each file.
            dry_run: If True, show what would be synced without syncing.

        Returns:
            SyncResult with success status and details.
        """
        if not self.remote_name:
            return SyncResult(
                success=False,
                files_synced=0,
                bytes_transferred=0,
                direction="download",
                error_message="No remote configured. Set remote_name first.",
            )

        remote_src = f"{self.remote_name}:{self.remote_path}"

        cmd = ["rclone", "copy", remote_src, str(local_path), "--verbose"]
        if dry_run:
            cmd.append("--dry-run")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            files_completed = 0
            output_lines = []

            for line in process.stderr:
                output_lines.append(line)

                if ": Copied" in line or "Would copy" in line:
                    files_completed += 1
                    match = re.search(r":\s*(.+?):\s*(?:Copied|Would copy)", line)
                    current_file = match.group(1) if match else ""
                    progress_callback(SyncProgress(
                        current_file=current_file,
                        files_completed=files_completed,
                    ))

            process.wait()

            if process.returncode == 0:
                output = "".join(output_lines)
                return SyncResult(
                    success=True,
                    files_synced=files_completed,
                    bytes_transferred=self._parse_bytes_transferred(output),
                    direction="download",
                )
            else:
                error_msg = "".join(output_lines).strip()
                return SyncResult(
                    success=False,
                    files_synced=0,
                    bytes_transferred=0,
                    direction="download",
                    error_message=error_msg or "Sync failed",
                )

        except FileNotFoundError:
            return SyncResult(
                success=False,
                files_synced=0,
                bytes_transferred=0,
                direction="download",
                error_message="rclone not found. Please install rclone.",
            )

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

        Handles two rclone output formats:
        1. Verbose per-file logs: "INFO  : file.pdf: Copied (new)"
        2. Stats summary: "Transferred: 3 files"

        Args:
            output: Output from rclone command.

        Returns:
            Number of files transferred or 0.
        """
        count = 0
        for line in output.split("\n"):
            # Count actual transfers: "INFO  : file.pdf: Copied (new)"
            if ": Copied" in line:
                count += 1
            # Count dry-run would-copy: "NOTICE: file.pdf: Would copy"
            elif "Would copy" in line:
                count += 1
            # Also handle stats summary format: "Transferred: 3 files"
            elif "transferred:" in line.lower() and "files" in line.lower():
                match = re.search(r"(\d+)\s*files?", line.lower())
                if match:
                    return int(match.group(1))
        return count

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
