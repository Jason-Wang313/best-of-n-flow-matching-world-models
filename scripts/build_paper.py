from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FINAL = PAPER / "final" / "best of n flow matching world models-v4.pdf"
FAILURE_LOG = ROOT / "docs" / "latex_failure.txt"
SCRATCH_EXTENSIONS = [".aux", ".log", ".out", ".toc", ".bbl", ".blg"]


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def compile_with_pdflatex() -> tuple[bool, str]:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        return False, "pdflatex not found on PATH"
    for ext in SCRATCH_EXTENSIONS:
        scratch = PAPER / f"main{ext}"
        if scratch.exists():
            scratch.unlink()
    logs: list[str] = []
    for _ in range(3):
        proc = run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], PAPER)
        logs.append(proc.stdout)
        logs.append(proc.stderr)
        if proc.returncode != 0:
            return False, "\n".join(logs)
    return True, "\n".join(logs)


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "final").mkdir(parents=True, exist_ok=True)
    ok, log = compile_with_pdflatex()
    if not ok:
        FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        FAILURE_LOG.write_text(log, encoding="utf-8")
        raise SystemExit(f"LaTeX build failed; wrote {FAILURE_LOG}")
    source_pdf = PAPER / "main.pdf"
    if not source_pdf.exists():
        raise SystemExit("LaTeX reported success but paper/main.pdf was not created")
    shutil.copy2(source_pdf, FINAL)
    if FAILURE_LOG.exists():
        FAILURE_LOG.unlink()
    print(f"wrote {FINAL}")


if __name__ == "__main__":
    main()
