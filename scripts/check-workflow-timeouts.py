#!/usr/bin/env python3
"""Verify every GitHub Actions job declares a 3-hour (180 minute) timeout.

Every job in each .github/workflows/*.y{a,}ml file must set
``timeout-minutes: 180`` so that no workflow run can exceed three hours
(GitHub Actions has no workflow-level timeout, only a per-job one, and its
default is 360 minutes). Jobs that call a reusable workflow (a job-level
``uses:``) are skipped, because GitHub Actions does not allow
``timeout-minutes`` on them.

Exits non-zero (so it can gate CI) if any job is missing the timeout or
sets a different value. Set ``WORKFLOWS_DIR`` to check a different
directory (used by the unit test).
"""

from __future__ import annotations

import glob
import os
import sys

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - only hit without PyYAML installed
    sys.exit("PyYAML is required to run this check: python3 -m pip install pyyaml")

# 3 hours. Keep in sync with the timeout-minutes set on every job.
REQUIRED_TIMEOUT_MINUTES = 180


def workflows_dir() -> str:
    override = os.environ.get("WORKFLOWS_DIR")
    if override:
        return override
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, ".github", "workflows")


def main() -> int:
    wf_dir = workflows_dir()
    files = sorted(
        glob.glob(os.path.join(wf_dir, "*.yaml"))
        + glob.glob(os.path.join(wf_dir, "*.yml"))
    )
    if not files:
        sys.exit(f"no workflow files found under {wf_dir}")

    violations: list[str] = []
    checked = 0
    skipped = 0
    for path in files:
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            try:
                doc = yaml.safe_load(fh)
            except yaml.YAMLError as exc:
                violations.append(f"{name}: YAML parse error: {exc}")
                continue
        jobs = (doc or {}).get("jobs") or {}
        for job_id, job in jobs.items():
            job = job or {}
            if "uses" in job:  # reusable workflow call: timeout-minutes not allowed
                skipped += 1
                print(f"skip  {name}:{job_id} (reusable workflow call)")
                continue
            checked += 1
            timeout = job.get("timeout-minutes")
            if timeout == REQUIRED_TIMEOUT_MINUTES:
                print(f"ok    {name}:{job_id}")
            elif timeout is None:
                violations.append(
                    f"{name}: job '{job_id}' has no timeout-minutes "
                    f"(want {REQUIRED_TIMEOUT_MINUTES})"
                )
            else:
                violations.append(
                    f"{name}: job '{job_id}' has timeout-minutes={timeout} "
                    f"(want {REQUIRED_TIMEOUT_MINUTES})"
                )

    print()
    if violations:
        print(
            f"FAIL: {len(violations)} job(s) not capped at "
            f"{REQUIRED_TIMEOUT_MINUTES} minutes (3h):"
        )
        for v in violations:
            print(f"  - {v}")
        return 1
    print(
        f"PASS: all {checked} job(s) set timeout-minutes: "
        f"{REQUIRED_TIMEOUT_MINUTES} ({skipped} reusable-workflow job(s) skipped)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
