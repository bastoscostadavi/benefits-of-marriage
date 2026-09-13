#!/usr/bin/env python3
"""Regenerate every figure in results/, in order.

    python scripts/run_all.py            # all of them
    python scripts/run_all.py 03 07      # just those

Takes a couple of minutes end to end; fig11 is most of it (it runs the model at
N = 10^6).
"""

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS = sorted(p for p in HERE.glob("fig*.py"))


def main(argv):
    wanted = argv[1:]
    chosen = [p for p in SCRIPTS
              if not wanted or any(w in p.name for w in wanted)]
    if not chosen:
        print(f"nothing matched {wanted}; available: {[p.name for p in SCRIPTS]}")
        return 1

    failures = []
    t_all = time.time()
    for path in chosen:
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(path)], cwd=HERE)
        status = "ok" if proc.returncode == 0 else "FAILED"
        if proc.returncode != 0:
            failures.append(path.name)
        print(f"--- {path.name}: {status} ({time.time() - t0:.1f}s)\n")

    print(f"=== {len(chosen) - len(failures)}/{len(chosen)} scripts ok "
          f"in {time.time() - t_all:.1f}s")
    if failures:
        print("    failed:", ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
