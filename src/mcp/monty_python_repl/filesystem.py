"""Filesystem policy for the Monty Python REPL."""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import pydantic_monty

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOST_WORKSPACE = PROJECT_ROOT / "workspace"
VIRTUAL_WORKSPACE_ROOT = PurePosixPath("/workspace")
ROOT_DIRECTORY = PurePosixPath("/")


class HostWorkspaceOSAccess(pydantic_monty.AbstractOS):
    """Map the virtual ``/workspace`` directory to a host directory."""

    def __init__(
        self,
        host_workspace_root: Path,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the workspace-backed OS adapter.

        Args:
            host_workspace_root: Local directory backing the virtual workspace.
            environ: Optional environment variables visible inside the sandbox.
        """
        self.host_workspace_root = host_workspace_root.resolve()
        self.host_workspace_root.mkdir(parents=True, exist_ok=True)
        self._environ = {
            "PWD": str(VIRTUAL_WORKSPACE_ROOT),
            "WORKSPACE": str(VIRTUAL_WORKSPACE_ROOT),
        }
        if environ:
            self._environ.update(
                {str(key): str(value) for key, value in environ.items()}
            )
        self._tracked_artifacts: set[str] = set()

    def begin_artifact_tracking(self) -> None:
        """Start a fresh execution-scoped artifact tracking window."""
        self._tracked_artifacts.clear()

    def record_virtual_artifact(self, path: PurePosixPath | str) -> None:
        """Record a changed workspace file by virtual path.

        Args:
            path (PurePosixPath | str): Virtual or relative workspace path.
        """
        normalized = self._normalize_virtual_path(PurePosixPath(str(path)))
        if normalized == ROOT_DIRECTORY or self.path_is_dir(normalized):
            return
        self._tracked_artifacts.add(str(normalized))

    def record_host_artifact(self, host_path: Path) -> None:
        """Record a changed workspace file by host path.

        Args:
            host_path (Path): Host path inside the workspace root.
        """
        self._tracked_artifacts.add(str(self.virtualize_host_path(host_path)))

    def consume_tracked_artifacts(self) -> list[str]:
        """Return tracked artifacts for the current execution and clear them."""
        artifacts = sorted(self._tracked_artifacts)
        self._tracked_artifacts.clear()
        return artifacts

    def _normalize_virtual_path(self, path: PurePosixPath) -> PurePosixPath:
        """Normalize a virtual path and keep it inside ``/workspace``."""
        candidate = PurePosixPath(path)
        if not candidate.is_absolute():
            candidate = VIRTUAL_WORKSPACE_ROOT / candidate

        normalized = PurePosixPath(posixpath.normpath(str(candidate)))
        if not str(normalized).startswith("/"):
            normalized = PurePosixPath("/") / normalized

        if normalized == ROOT_DIRECTORY:
            return normalized
        if normalized == VIRTUAL_WORKSPACE_ROOT or normalized.is_relative_to(
            VIRTUAL_WORKSPACE_ROOT
        ):
            return normalized
        raise PermissionError(f"Path {normalized!s} is outside the /workspace sandbox.")

    def _is_virtual_root(self, path: PurePosixPath) -> bool:
        """Return ``True`` when the path is the virtual root directory."""
        return path == ROOT_DIRECTORY

    def _to_host_path(self, path: PurePosixPath) -> Path:
        """Translate a virtual path to the host workspace."""
        normalized = self._normalize_virtual_path(path)
        if normalized == ROOT_DIRECTORY:
            raise PermissionError(
                "The virtual root does not map to a writable host path."
            )

        relative = normalized.relative_to(VIRTUAL_WORKSPACE_ROOT)
        host_path = (self.host_workspace_root / Path(relative.as_posix())).resolve(
            strict=False
        )
        host_path.relative_to(self.host_workspace_root)
        return host_path

    def to_host_path(self, path: PurePosixPath | str) -> Path:
        """Translate a virtual or relative path to the host workspace.

        Args:
            path (PurePosixPath | str): Virtual or relative workspace path.

        Returns:
            Path: Resolved host path inside the workspace root.
        """
        return self._to_host_path(PurePosixPath(str(path)))

    def virtualize_host_path(self, host_path: Path) -> PurePosixPath:
        """Convert a host workspace path back into a virtual sandbox path."""
        relative = host_path.resolve(strict=False).relative_to(self.host_workspace_root)
        if str(relative) == ".":
            return VIRTUAL_WORKSPACE_ROOT
        return VIRTUAL_WORKSPACE_ROOT / PurePosixPath(relative.as_posix())

    def path_exists(self, path: PurePosixPath) -> bool:
        """Check whether a path exists inside the virtual filesystem."""
        normalized = self._normalize_virtual_path(path)
        if self._is_virtual_root(normalized):
            return True
        return self._to_host_path(normalized).exists()

    def path_is_file(self, path: PurePosixPath) -> bool:
        """Check whether a path is a regular file."""
        normalized = self._normalize_virtual_path(path)
        if self._is_virtual_root(normalized):
            return False
        return self._to_host_path(normalized).is_file()

    def path_is_dir(self, path: PurePosixPath) -> bool:
        """Check whether a path is a directory."""
        normalized = self._normalize_virtual_path(path)
        if self._is_virtual_root(normalized):
            return True
        return self._to_host_path(normalized).is_dir()

    def path_is_symlink(self, path: PurePosixPath) -> bool:
        """Report whether a path is a symlink."""
        normalized = self._normalize_virtual_path(path)
        if self._is_virtual_root(normalized):
            return False
        return self._to_host_path(normalized).is_symlink()

    def path_read_text(self, path: PurePosixPath) -> str:
        """Read a workspace file as UTF-8 text."""
        return self._to_host_path(path).read_text(encoding="utf-8")

    def path_read_bytes(self, path: PurePosixPath) -> bytes:
        """Read a workspace file as bytes."""
        return self._to_host_path(path).read_bytes()

    def path_write_text(self, path: PurePosixPath, data: str) -> int:
        """Write UTF-8 text into the workspace."""
        host_path = self._to_host_path(path)
        if host_path.exists() and host_path.is_dir():
            raise IsADirectoryError(f"[Errno 21] Is a directory: {str(path)!r}")
        if not host_path.parent.exists():
            raise FileNotFoundError(
                f"[Errno 2] No such file or directory: {str(path)!r}"
            )
        host_path.write_text(data, encoding="utf-8")
        self.record_virtual_artifact(path)
        return len(data)

    def path_write_bytes(self, path: PurePosixPath, data: bytes) -> int:
        """Write binary data into the workspace."""
        host_path = self._to_host_path(path)
        if host_path.exists() and host_path.is_dir():
            raise IsADirectoryError(f"[Errno 21] Is a directory: {str(path)!r}")
        if not host_path.parent.exists():
            raise FileNotFoundError(
                f"[Errno 2] No such file or directory: {str(path)!r}"
            )
        host_path.write_bytes(data)
        self.record_virtual_artifact(path)
        return len(data)

    def path_mkdir(self, path: PurePosixPath, parents: bool, exist_ok: bool) -> None:
        """Create a directory inside the workspace."""
        normalized = self._normalize_virtual_path(path)
        if self._is_virtual_root(normalized):
            if exist_ok:
                return
            raise FileExistsError(f"[Errno 17] File exists: {str(path)!r}")
        self._to_host_path(normalized).mkdir(parents=parents, exist_ok=exist_ok)

    def path_unlink(self, path: PurePosixPath) -> None:
        """Delete a file from the workspace."""
        self._to_host_path(path).unlink()

    def path_rmdir(self, path: PurePosixPath) -> None:
        """Remove an empty directory from the workspace."""
        normalized = self._normalize_virtual_path(path)
        if self._is_virtual_root(normalized):
            raise PermissionError("The virtual root directory cannot be removed.")
        self._to_host_path(normalized).rmdir()

    def path_iterdir(self, path: PurePosixPath) -> list[PurePosixPath]:
        """List directory contents as virtual paths."""
        normalized = self._normalize_virtual_path(path)
        if self._is_virtual_root(normalized):
            return [VIRTUAL_WORKSPACE_ROOT]
        return [
            self.virtualize_host_path(child)
            for child in sorted(
                self._to_host_path(normalized).iterdir(),
                key=lambda item: item.name,
            )
        ]

    def path_stat(self, path: PurePosixPath) -> pydantic_monty.StatResult:
        """Return stat information for a workspace path."""
        normalized = self._normalize_virtual_path(path)
        if self._is_virtual_root(normalized):
            return pydantic_monty.StatResult.dir_stat()

        host_path = self._to_host_path(normalized)
        stat_result = host_path.stat()
        if host_path.is_dir():
            return pydantic_monty.StatResult.dir_stat(mode=stat_result.st_mode)
        return pydantic_monty.StatResult.file_stat(
            size=stat_result.st_size,
            mode=stat_result.st_mode,
        )

    def path_rename(self, path: PurePosixPath, target: PurePosixPath) -> None:
        """Rename a file or directory inside the workspace."""
        normalized = self._normalize_virtual_path(path)
        if self._is_virtual_root(normalized):
            raise PermissionError("The virtual root directory cannot be renamed.")
        target_normalized = self._normalize_virtual_path(target)
        self._to_host_path(normalized).rename(self._to_host_path(target_normalized))
        if not self.path_is_dir(target_normalized):
            self.record_virtual_artifact(target_normalized)

    def path_resolve(self, path: PurePosixPath) -> str:
        """Resolve a path into its canonical virtual form."""
        return str(self._normalize_virtual_path(path))

    def path_absolute(self, path: PurePosixPath) -> str:
        """Convert a path into an absolute virtual workspace path."""
        return str(self._normalize_virtual_path(path))

    def getenv(self, key: str, default: str | None = None) -> str | None:
        """Fetch a virtual environment variable value."""
        return self._environ.get(key, default)

    def get_environ(self) -> dict[str, str]:
        """Return the virtual environment dictionary."""
        return dict(self._environ)
