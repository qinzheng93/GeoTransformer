#!/usr/bin/env python3
"""Generate a Markdown diff report between two git revisions.

Example:
    python scripts/commit_diff_report.py --from HEAD~1 --to HEAD \
        --output docs/commit-diff-latest.md

Defaults:
    --from HEAD~1
    --to   HEAD
    --output docs/commit-diff-<from>..<to>.md (created under repo root)
"""
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Simple helper to run git commands and return stdout text

def run_git(args: List[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def get_repo_root() -> Path:
    root = run_git(["rev-parse", "--show-toplevel"], Path.cwd())
    return Path(root)


def get_commit_info(rev: str, cwd: Path) -> Dict[str, str]:
    fmt = "%h%n%H%n%an%n%ad%n%s"
    out = run_git(["show", "-s", f"--format={fmt}", rev], cwd)
    short, full, author, date, subject = out.split("\n", 4)
    return {
        "short": short,
        "full": full,
        "author": author,
        "date": date,
        "subject": subject,
    }


def get_shortstat(rev_from: str, rev_to: str, cwd: Path) -> Dict[str, int]:
    line = run_git(["diff", "--shortstat", f"{rev_from}..{rev_to}"], cwd)
    # Example: "15 files changed, 2155 insertions(+), 823 deletions(-)"
    files = inserts = deletes = 0
    parts = line.replace(",", "").split()
    for i, p in enumerate(parts):
        if p == "files":
            files = int(parts[i - 1])
        elif p == "insertions(+)":
            inserts = int(parts[i - 1])
        elif p == "deletions(-)":
            deletes = int(parts[i - 1])
    return {"files": files, "insertions": inserts, "deletions": deletes}


def get_stat(rev_from: str, rev_to: str, cwd: Path) -> List[str]:
    out = run_git(["diff", "--stat", f"{rev_from}..{rev_to}"], cwd)
    return [line for line in out.splitlines() if line.strip()]


def get_name_status(rev_from: str, rev_to: str, cwd: Path) -> List[str]:
    out = run_git(["diff", "--name-status", f"{rev_from}..{rev_to}"], cwd)
    return [line for line in out.splitlines() if line.strip()]


def get_numstat(rev_from: str, rev_to: str, cwd: Path) -> List[Tuple[str, str, str]]:
    out = run_git(["diff", "--numstat", f"{rev_from}..{rev_to}"], cwd)
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            add, delete, path = parts[0], parts[1], parts[2]
            rows.append((add, delete, path))
    return rows


def build_markdown(
    rev_from: str,
    rev_to: str,
    from_info: Dict[str, str],
    to_info: Dict[str, str],
    shortstat: Dict[str, int],
    stat_lines: List[str],
    name_status: List[str],
    numstat: List[Tuple[str, str, str]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Commit Diff: {from_info['short']}..{to_info['short']}")
    lines.append("")
    lines.append(f"- Range: `{rev_from}` → `{rev_to}`")
    lines.append(f"- Date: {to_info['date']}")
    lines.append("")
    lines.append("## Commits")
    lines.append(
        f"- Previous: `{from_info['short']}` — {from_info['subject']} (Author: {from_info['author']}; {from_info['date']})"
    )
    lines.append(
        f"- Current:  `{to_info['short']}` — {to_info['subject']} (Author: {to_info['author']}; {to_info['date']})"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Files changed: {shortstat['files']}")
    lines.append(f"- Insertions: {shortstat['insertions']}")
    lines.append(f"- Deletions: {shortstat['deletions']}")
    lines.append("")
    lines.append("## File Changes (git diff --stat)")
    lines.extend([f"- {line}" for line in stat_lines])
    lines.append("")
    lines.append("## Name Status (git diff --name-status)")
    lines.extend([f"- {line}" for line in name_status])
    lines.append("")
    lines.append("## Numstat (additions\tdeletions\tfile)")
    for add, delete, path in numstat:
        lines.append(f"- {add}\t{delete}\t{path}")
    lines.append("")
    lines.append("---")
    lines.append(
        "Generated automatically by scripts/commit_diff_report.py using git diff stats for the specified range."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown diff report between two git revisions.")
    parser.add_argument("--from", dest="rev_from", default="HEAD~1", help="Old/base revision (default: HEAD~1)")
    parser.add_argument("--to", dest="rev_to", default="HEAD", help="New revision (default: HEAD)")
    parser.add_argument(
        "--output",
        dest="output",
        default=None,
        help="Output Markdown file path. Defaults to docs/commit-diff-<from>..<to>.md under repo root.",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    rev_from = args.rev_from
    rev_to = args.rev_to

    from_info = get_commit_info(rev_from, repo_root)
    to_info = get_commit_info(rev_to, repo_root)
    shortstat = get_shortstat(rev_from, rev_to, repo_root)
    stat_lines = get_stat(rev_from, rev_to, repo_root)
    name_status = get_name_status(rev_from, rev_to, repo_root)
    numstat = get_numstat(rev_from, rev_to, repo_root)

    default_name = f"commit-diff-{from_info['short']}..{to_info['short']}.md"
    output_path = Path(args.output) if args.output else repo_root / "docs" / default_name
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = build_markdown(rev_from, rev_to, from_info, to_info, shortstat, stat_lines, name_status, numstat)
    output_path.write_text(markdown)
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
