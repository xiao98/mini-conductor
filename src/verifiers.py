"""Deterministic verifiers for Math (sympy), Code (isolated subprocess sandbox), and GPQA (MCQ)."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


# ---------------------------------------------------------------------------
# Math Verifier
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """Extract the content inside the last `\\boxed{...}`, properly handling nested braces."""
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    return text[start : pos - 1].strip()


def _normalize_latex_expr(expr: str) -> str:
    s = expr.strip()
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\%", "%").replace("\\cdot", "*").replace("\\times", "*")
    # `5.4 \text{ cents}` -> `5.4` (unit dropped); a pure-text answer like `\text{east}` is kept.
    without_text = re.sub(r"\\text\{[^}]*\}", "", s)
    s = without_text if without_text.strip() else re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\{?\\circ\}?", "", s).replace("\\degree", "")   # 120^\circ -> 120
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)             # \frac43 -> \frac{4}{3}
    s = re.sub(r"\s+", "", s)                                        # (15, -29) == (15,-29)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):               # 58,500 -> 58500
        s = s.replace(",", "")
    return s.rstrip(".")


def _latex_to_sympy_str(s: str) -> str:
    """Lightweight LaTeX -> SymPy converter for fractions, sqrt, and exponents."""
    out = _normalize_latex_expr(s)
    # Convert \frac{a}{b} iteratively (handles simple nesting)
    frac_re = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    for _ in range(5):
        new_out = frac_re.sub(r"((\1)/(\2))", out)
        if new_out == out:
            break
        out = new_out
    # Convert \sqrt{a}
    sqrt_re = re.compile(r"\\sqrt\{([^{}]+)\}")
    for _ in range(5):
        new_out = sqrt_re.sub(r"sqrt(\1)", out)
        if new_out == out:
            break
        out = new_out
    out = out.replace("\\pi", "pi").replace("^", "**")
    return out


def verify_math(prediction: str, ground_truth: str) -> tuple[bool, str]:
    """Verify mathematical equivalence using exact normalized string match + SymPy."""
    pred_ans = extract_boxed(prediction)
    if pred_ans is None:
        # Fallback: check "Answer: ..." or last non-empty line
        m = re.findall(r"(?:final answer|answer)\s*[:=]\s*([^\n]+)", prediction, flags=re.IGNORECASE)
        if m:
            pred_ans = m[-1].strip()
        else:
            lines = [ln.strip() for ln in prediction.strip().splitlines() if ln.strip()]
            pred_ans = lines[-1] if lines else ""

    gt_boxed = extract_boxed(ground_truth)
    gt_ans = gt_boxed if gt_boxed is not None else ground_truth.strip()

    norm_pred = _normalize_latex_expr(pred_ans)
    norm_gt = _normalize_latex_expr(gt_ans)

    if not norm_pred:
        return False, "empty_prediction"
    if norm_pred == norm_gt:
        return True, f"exact_match:{norm_pred}"

    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transforms = standard_transformations + (implicit_multiplication_application,)
        e_pred = parse_expr(_latex_to_sympy_str(norm_pred), transformations=transforms)
        e_gt = parse_expr(_latex_to_sympy_str(norm_gt), transformations=transforms)

        diff = sp.simplify(e_pred - e_gt)
        if diff == 0:
            return True, f"sympy_exact:{norm_pred}=={norm_gt}"
        # Numeric fallback for float representations
        if abs(complex(e_pred.evalf()) - complex(e_gt.evalf())) < 1e-6:
            return True, f"sympy_numeric:{norm_pred}=={norm_gt}"
    except Exception:
        pass

    return False, f"mismatch:pred={norm_pred}|gt={norm_gt}"


# ---------------------------------------------------------------------------
# Code Verifier (Isolated Subprocess Sandbox)
# ---------------------------------------------------------------------------

_PY_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    """Extract the last Python fenced block, or fall back to raw text if code-like."""
    matches = _PY_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def verify_code(
    prediction: str,
    ground_truth_tests: str,
    *,
    metadata: dict[str, Any] | None = None,
    timeout_s: float = 5.0,
) -> tuple[bool, str]:
    """Execute generated Python code against unit tests or stdin/stdout cases in an isolated subprocess."""
    code = extract_python_code(prediction)
    if not code:
        return False, "empty_code"

    meta = metadata or {}
    io_cases = meta.get("io_tests")  # Optional list of {"input": "...", "output": "..."}

    with tempfile.TemporaryDirectory(prefix="mc_sandbox_") as tmpdir:
        script_path = Path(tmpdir) / "solution.py"

        if io_cases:
            script_path.write_text(code, encoding="utf-8")
            for idx, case in enumerate(io_cases):
                try:
                    proc = subprocess.run(
                        [sys.executable, "-I", str(script_path)],
                        input=str(case["input"]),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout_s,
                        cwd=tmpdir,
                    )
                except subprocess.TimeoutExpired:
                    return False, f"timeout_on_case_{idx}"
                if proc.returncode != 0:
                    return False, f"runtime_error_case_{idx}:{proc.stderr.strip()[:200]}"
                actual = proc.stdout.strip()
                expected = str(case["output"]).strip()
                if actual != expected:
                    return False, f"wrong_output_case_{idx}:got={actual[:80]}|exp={expected[:80]}"
            return True, f"passed_{len(io_cases)}_io_cases"

        # Unit-test assertion mode (MBPP / HumanEval / KodCode style)
        combined = f"{code}\n\n# --- Hidden Unit Tests ---\n{ground_truth_tests}\n"
        script_path.write_text(combined, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                cwd=tmpdir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if proc.returncode == 0:
            return True, "tests_passed"
        err = (proc.stderr or proc.stdout).strip().splitlines()
        summary = err[-1][:200] if err else f"exit_{proc.returncode}"
        return False, f"test_failed:{summary}"


# ---------------------------------------------------------------------------
# GPQA / Multiple-Choice Verifier
# ---------------------------------------------------------------------------

def verify_gpqa(prediction: str, ground_truth: str) -> tuple[bool, str]:
    """Verify multiple-choice option letter (A/B/C/D)."""
    gt = ground_truth.strip().upper()
    m_gt = re.search(r"([A-D])", gt)
    if not m_gt:
        return False, f"invalid_gt:{ground_truth}"
    gt_letter = m_gt.group(1)

    # Check high-precision patterns first
    patterns = [
        r"(?:final\s+answer|answer)\s*[:=\-]\s*\(?([A-D])\)?\b",
        r"\\boxed\{\s*\(?([A-D])\)?\s*\}",
        r"(?:option|choice)\s+\(?([A-D])\)?\b",
        r"(?:^|\s)\(([A-D])\)(?:\s|$|\.)",
    ]
    pred_letter: str | None = None
    for pat in patterns:
        matches = re.findall(pat, prediction, flags=re.IGNORECASE)
        if matches:
            pred_letter = matches[-1].upper()
            break

    if pred_letter is None:
        tail = prediction.strip()[-32:]
        m_tail = re.findall(r"\b([A-D])\b", tail)
        if m_tail:
            pred_letter = m_tail[-1].upper()

    if pred_letter is None:
        return False, "no_choice_extracted"
    if pred_letter == gt_letter:
        return True, f"mcq_match:{pred_letter}"
    return False, f"mcq_mismatch:pred={pred_letter}|gt={gt_letter}"


# ---------------------------------------------------------------------------
# Unified Dispatcher
# ---------------------------------------------------------------------------

def verify_answer(
    task_type: str,
    prediction: str,
    ground_truth: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Dispatch verification by task type."""
    t = task_type.lower().strip()
    if t == "math":
        return verify_math(prediction, ground_truth)
    if t == "code":
        return verify_code(prediction, ground_truth, metadata=metadata)
    if t in {"gpqa", "mcq", "science"}:
        return verify_gpqa(prediction, ground_truth)
    raise ValueError(f"Unsupported task_type: {task_type}")
