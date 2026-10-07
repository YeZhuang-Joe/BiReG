#!/usr/bin/env python3
"""Check the candidate's two-line output without repairing or scoring it.

Usage: python check_bireg_plan.py response.txt --max-regions 7
Uses only the Python standard library. This does not call an LLM or a GPU.
"""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import sys


def check_plan(text, max_regions=7):
    """Return a parsed plan, or raise ValueError with a structural error.

    This deliberately accepts the candidate's canonical grammar, not every
    legacy ratio form. It does not verify semantic fidelity or token length.
    """
    if not isinstance(max_regions, int) or isinstance(max_regions, bool) or max_regions < 1:
        raise ValueError("max_regions must be a positive integer")
    # A final newline is allowed; blank lines, prose and code fences are not.
    lines = text.splitlines()
    if len(lines) != 2:
        raise ValueError("expected exactly two output lines")
    ratio_label = "Final split ratio: "
    prompt_label = "Regional Prompt: "
    if not lines[0].startswith(ratio_label) or not lines[1].startswith(prompt_label):
        raise ValueError("incorrect output labels or label order")
    ratio = lines[0][len(ratio_label):]
    regional_prompt = lines[1][len(prompt_label):]
    number = r"(?:0\.[0-9]+|1\.0+)"
    if not re.fullmatch(number + r"(?:[,;]" + number + r")*", ratio):
        raise ValueError("ratios must be decimals in (0, 1] with ASCII separators and no spaces")
    numeric_rows = [[Decimal(v) for v in row.split(",")] for row in ratio.split(";")]
    if any(v <= 0 or v > 1 for row in numeric_rows for v in row):
        raise ValueError("every ratio must be greater than zero and at most one")
    if len(numeric_rows) == 1:
        heights = [Decimal(1)]
        widths = numeric_rows
    else:
        if any(len(row) < 2 for row in numeric_rows):
            raise ValueError("multi-row form requires an explicit height and width(s) in every row")
        heights = [row[0] for row in numeric_rows]
        widths = [row[1:] for row in numeric_rows]
    if sum(heights) != Decimal(1):
        raise ValueError("row heights must sum exactly to one")
    if any(sum(row) != Decimal(1) for row in widths):
        raise ValueError("region widths within every row must sum exactly to one")
    region_count = sum(len(row) for row in widths)
    if region_count > max_regions:
        raise ValueError("region count exceeds configured max_regions")
    segments = regional_prompt.split(" BREAK ")
    if len(segments) != region_count:
        raise ValueError("number of regional descriptions does not match ratio region count")
    if any(not part or not part.strip() or part != part.strip() for part in segments):
        raise ValueError("empty description or incorrect spacing around a description")
    if any("BREAK" in part for part in segments):
        raise ValueError("reserved BREAK token used without its exact delimiter spacing")
    return {
        "split_ratio": ratio,
        "regional_prompt": regional_prompt,
        "region_count": region_count,
        "row_heights": [str(v) for v in heights],
        "row_widths": [[str(v) for v in row] for row in widths],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path)
    parser.add_argument("--max-regions", type=int, default=7)
    args = parser.parse_args()
    try:
        plan = check_plan(args.response.read_text(encoding="utf-8"), args.max_regions)
    except (OSError, UnicodeError, ValueError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "structurally_valid", **plan}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
