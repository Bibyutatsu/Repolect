"""
Repolect — Git Utilities
Change detection is the key to fast incremental sync.
"""
 
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path
 
 
def find_repo_root(start_path: str | Path = ".") -> str | None:
    """Walk up the directory tree looking for a .git directory."""
    current = Path(start_path).resolve()
    while True:
        if (current / ".git").exists():
            return str(current)
        parent = current.parent
        if parent == current:
            return None
        current = parent
 
 
def get_current_branch(repo_root: str | Path) -> str:
    """Return the current branch name, or 'HEAD' for detached state."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            return branch if branch else "HEAD"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "HEAD"
 
 
def get_all_branches(repo_root: str | Path) -> list[str]:
    """Return all local branch names."""
    try:
        result = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return []
 
 
def checkout_branch(repo_root: str | Path, branch: str) -> bool:
    """Checkout a branch. Returns True on success."""
    try:
        result = subprocess.run(
            ["git", "checkout", branch],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
 
 
def get_current_commit(repo_root: str | Path) -> str:
    """Return the current HEAD commit hash, or 'no-git' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "no-git"
 
 
def get_repo_name(repo_root: str | Path) -> str:
    """Try to get a meaningful repo name (remote name or directory name)."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract repo name from URL: https://github.com/user/repo.git → repo
            name = url.rstrip("/").rstrip(".git").split("/")[-1]
            if name:
                return name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return Path(repo_root).name
 
 
def get_file_hash(file_path: str | Path) -> str:
    """SHA256 of file contents — used for change detection."""
    try:
        content = Path(file_path).read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]
    except (IOError, OSError):
        return ""
 
 
def get_changed_files(repo_root: str | Path, since_commit: str) -> list[str]:
    """
    Return list of files changed since a given commit.
    Falls back to all files if git is unavailable.
    """
    if since_commit == "no-git":
        return []
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", since_commit, "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return [
                str(Path(repo_root) / f.strip())
                for f in result.stdout.strip().split("\n")
                if f.strip()
            ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return []
 
 
def get_changed_line_ranges(
    repo_root: str | Path,
    ref: str = "HEAD~1",
    committed_only: bool = False,
) -> dict[str, list[tuple[int, int]]]:
    """Return {relative_file_path: [(start_line, end_line), ...]} for all changed hunks.
 
    Parses ``git diff --unified=0`` to extract exact changed line ranges in the
    new (current) version of each file.
 
    Args:
        ref: Git ref to diff against (default: HEAD~1).
        committed_only: If True, diff ref..HEAD (committed only).
                        If False (default), diff ref vs working tree
                        (includes uncommitted changes).
    """
    import re as _re
    cmd = ["git", "diff", "--unified=0", ref]
    if committed_only:
        cmd.append("HEAD")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return {}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
 
    changes: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
 
    for line in result.stdout.split("\n"):
        m = _re.match(r"^diff --git a/.+? b/(.+?)$", line)
        if m:
            current_file = m.group(1)
            if current_file not in changes:
                changes[current_file] = []
            continue
        m = _re.match(r"^@@ .+ \+(\d+)(?:,(\d+))? @@", line)
        if m and current_file:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) else 1
            end = start + max(count - 1, 0)
            changes[current_file].append((start, end))
 
    return changes
 
 
def get_git_log_frequency(repo_root: str | Path, file_path: str | Path) -> int:
    """
    Count how many commits touched this file — used as 'change_frequency' signal.
    High frequency = hot/important/unstable file.
    """
    rel_path = Path(file_path).relative_to(Path(repo_root))
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--", str(rel_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().split("\n") if l])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0
 
 
def is_git_repo(path: str | Path = ".") -> bool:
    """Returns True if .git/ exists in path or any ancestor."""
    return find_repo_root(path) is not None
 
 
def detect_repo_root(start_path: str | Path = ".") -> str:
    """
    Return a usable repo root — works with or without git.
    If inside a git repo, returns the git root.
    Otherwise returns the resolved absolute path of start_path.
    """
    git_root = find_repo_root(start_path)
    if git_root:
        return git_root
    return str(Path(start_path).resolve())
 
 
def get_file_authors(repo_root: str | Path, file_path: str | Path) -> list[str]:
    """Return list of unique authors who've touched this file."""
    rel_path = Path(file_path).relative_to(Path(repo_root))
    try:
        result = subprocess.run(
            ["git", "log", "--format=%an", "--", str(rel_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return list(set(l.strip() for l in result.stdout.strip().split("\n") if l.strip()))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return []
 