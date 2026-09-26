"""Make the upstream `dgl` package importable inside this environment.

DGL 2.3.0 predates torchdata 0.10, which removed `torchdata.datapipes`; its own
conda recipe happily resolves to torchdata 0.11, and `import dgl` then dies
inside `dgl.graphbolt.base`. The fix is the last torchdata that still ships the
datapipes, installed from PyPI over the conda one. Run once; the guard makes
every later `pixi run test` a no-op.
"""

from __future__ import annotations

import subprocess
import sys

TORCHDATA = "torchdata==0.7.1"


def compatible() -> bool:
    try:
        import torchdata.datapipes.iter  # noqa: F401
    except Exception:
        return False
    return True


def main() -> int:
    if compatible():
        return 0
    print("installing {} so that `import dgl` works".format(TORCHDATA))
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
         TORCHDATA],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print((proc.stderr or proc.stdout).strip()[-2000:], file=sys.stderr)
        print(
            "warning: could not install {}; the parity tests that import dgl "
            "will be skipped".format(TORCHDATA),
            file=sys.stderr,
        )
        return 0
    if not compatible():
        print(
            "warning: {} did not restore torchdata.datapipes".format(TORCHDATA),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
