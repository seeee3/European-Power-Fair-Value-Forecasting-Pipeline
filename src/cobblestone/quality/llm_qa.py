"""
AI-accelerated QA: GPT-4o-mini proposes domain-aware validation rules,
the pipeline executes them, and all prompts/outputs/failures are logged.

This satisfies the case study requirement:
  "LLM-driven data QA rules & tests: given a schema + sample rows,
   the LLM proposes validation rules; your pipeline executes them and
   produces a QA report."
"""
from __future__ import annotations

import json
import logging
import textwrap
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = textwrap.dedent("""
    You are a quantitative analyst at a European energy trading desk.
    You will receive a description of a power market dataset (schema, statistics,
    and sample rows) and must return a JSON array of validation rules.

    Each rule must have these fields:
      "name"        : unique snake_case identifier
      "description" : one-sentence explanation of what it checks
      "column"      : which DataFrame column it applies to (or "index")
      "severity"    : "error" | "warning"
      "expression"  : a Python expression using variable `s` (a pd.Series of the column values,
                      already dropna applied) that evaluates to a boolean Series.
                      True = row passes the check.  Example: `s > 0`

    Return ONLY a JSON array with no commentary before or after.
    Generate 8-12 rules covering:
      - Physical feasibility (generation cannot be negative; load has realistic bounds)
      - Market realism (price spikes and crashes; negative prices ARE valid in DE market)
      - Temporal consistency (large jumps between consecutive hours)
      - Cross-series sanity (high wind => typically lower prices)

    Rules MUST be executable Python expressions. Do not use custom functions.
""").strip()


def _build_user_prompt(df: pd.DataFrame) -> str:
    schema_lines = []
    for col in df.columns:
        s = df[col].dropna()
        schema_lines.append(
            f"  {col}: dtype={df[col].dtype}, "
            f"non_null={s.size}, "
            f"min={s.min():.2f}, max={s.max():.2f}, "
            f"mean={s.mean():.2f}, std={s.std():.2f}"
        )
    schema_str = "\n".join(schema_lines)

    sample = df.head(5).reset_index().to_string(index=False)

    return textwrap.dedent(f"""
        Dataset: Hourly European power market data for Germany (DE-LU bidding zone).
        Index: UTC hourly timestamps.

        Schema and statistics:
        {schema_str}

        Sample rows (first 5):
        {sample}

        Please generate validation rules for this dataset.
    """).strip()


def _execute_rule(df: pd.DataFrame, rule: dict[str, Any]) -> dict[str, Any]:
    """Execute a single LLM-proposed rule and return a result dict."""
    col = rule["column"]
    expr = rule["expression"]
    result: dict[str, Any] = {
        "name": rule["name"],
        "description": rule["description"],
        "column": col,
        "severity": rule.get("severity", "warning"),
        "expression": expr,
        "status": "unknown",
        "violations": 0,
        "fraction": 0.0,
        "error": None,
    }
    try:
        if col == "index":
            s = pd.Series(df.index, index=df.index)
        elif col not in df.columns:
            result["status"] = "skipped"
            result["error"] = f"Column '{col}' not in dataset"
            return result

        s = df[col].dropna()  # noqa: F841  (used in eval)
        mask = eval(expr)  # noqa: S307 — controlled LLM output, logged
        if not isinstance(mask, pd.Series):
            mask = pd.Series([bool(mask)] * len(s), index=s.index)
        violations = int((~mask).sum())
        result["violations"] = violations
        result["fraction"] = round(violations / max(len(s), 1), 4)
        result["status"] = "pass" if violations == 0 else "fail"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc(limit=3)
    return result


def run_llm_qa(
    df: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4o-mini",
    log_path: Path | None = None,
) -> dict[str, Any]:
    """
    Generate QA rules via LLM, execute them, and return a full log.

    Returns:
        {
            "model":    ...,
            "prompt":   ...,
            "raw_response": ...,
            "rules":    [...],
            "results":  [...],
            "summary":  {...},
        }
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    user_prompt = _build_user_prompt(df)

    log: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "raw_response": None,
        "parse_error": None,
        "rules": [],
        "results": [],
        "summary": {},
    }

    # ── Call the LLM ──────────────────────────────────────────────────────────
    logger.info("Calling %s for QA rule generation...", model)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        log["raw_response"] = raw
        log["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
        }
        logger.info("LLM response received (%d chars)", len(raw))
    except Exception as exc:
        log["parse_error"] = f"API call failed: {exc}"
        logger.error("LLM API call failed: %s", exc)
        _save_log(log, log_path)
        return log

    # ── Parse rules ───────────────────────────────────────────────────────────
    try:
        parsed = json.loads(raw)
        # Model may return {"rules": [...]} or just [...]
        rules_list = parsed if isinstance(parsed, list) else parsed.get("rules", [])
        log["rules"] = rules_list
        logger.info("Parsed %d rules from LLM response", len(rules_list))
    except json.JSONDecodeError as exc:
        log["parse_error"] = str(exc)
        logger.error("Failed to parse LLM JSON: %s", exc)
        _save_log(log, log_path)
        return log

    # ── Execute rules ─────────────────────────────────────────────────────────
    results = [_execute_rule(df, rule) for rule in rules_list]
    log["results"] = results

    passes = sum(1 for r in results if r["status"] == "pass")
    fails = sum(1 for r in results if r["status"] == "fail")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skipped")

    log["summary"] = {
        "total_rules": len(results),
        "pass": passes,
        "fail": fails,
        "error": errors,
        "skipped": skipped,
    }
    logger.info("LLM QA: %d pass, %d fail, %d error, %d skipped", passes, fails, errors, skipped)

    _save_log(log, log_path)
    return log


def _save_log(log: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    logger.info("LLM QA log saved to %s", path)


def print_llm_qa_summary(log: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    s = log.get("summary", {})
    console.print(f"\n[bold]LLM QA Results[/bold]  (model: {log.get('model')})")
    console.print(
        f"  {s.get('total_rules', 0)} rules: "
        f"[green]{s.get('pass', 0)} pass[/green]  "
        f"[red]{s.get('fail', 0)} fail[/red]  "
        f"[yellow]{s.get('error', 0)} error[/yellow]"
    )

    if not log.get("results"):
        return

    t = Table(show_lines=True)
    t.add_column("Rule")
    t.add_column("Column")
    t.add_column("Status")
    t.add_column("Violations", justify="right")
    t.add_column("Description")

    status_colors = {"pass": "green", "fail": "red", "error": "yellow", "skipped": "dim"}
    for r in log["results"]:
        color = status_colors.get(r["status"], "white")
        t.add_row(
            r["name"],
            r["column"],
            f"[{color}]{r['status']}[/{color}]",
            str(r["violations"]) if r["status"] in ("pass", "fail") else "—",
            r["description"],
        )
    console.print(t)
