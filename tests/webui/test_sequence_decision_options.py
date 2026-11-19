"""Round-trip test between a decision's candidate values and the field that edits them.

A sequence block declares the configuration axes it puts under test, each with
the candidate values the experiment compares, and the block editor edits that
list as one comma-separated field. A dict value renders with its braces and a
tuple or list candidate is typed in brackets, so either carries commas of its
own: splitting the field on every comma would tear ``(-46.63, -23.55)`` into two
candidates. The field is parsed by ``PythonLiteral.split`` instead, which breaks
only at the commas outside brackets, braces and quotes.

The parsing lives in the browser, so this module runs that real front-end helper
under node, against the strings the Python renderer produces for dict values,
against bracketed and quoted candidates, and against plain scalar candidates.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from config_value_render import ConfigValueRenderer

HARNESS = Path(__file__).with_name("literal_split.mjs")

DICT_VALUES = [
    {"distance_weight": 1.5, "unassigned_penalty_weight": 1.0},
    {"profile": "car", "geometry": True},
]

TYPED = ["(-46.63, -23.55)", "[4, 8, 16]", "'insertions, full'"]


def _split(fields: list, tmp_path: Path) -> list:
    """Returns each candidates field split into candidates by the browser's own helper.

    Args:
        fields   : Candidate fields as the block editor renders them, one comma-separated string per decision.
        tmp_path : Directory the harness payload is written into.

    Returns:
        One candidate list per field, in the order the fields were given.
    """
    payload = tmp_path / "fields.json"
    payload.write_text(json.dumps(fields), encoding="utf-8")

    result = subprocess.run([shutil.which("node"), str(HARNESS), str(payload)], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the front-end literal helper")
def test_rendered_dicts_and_bracketed_candidates_stay_one_candidate_each(tmp_path):
    """Dict values rendered by the launch form, bracketed tuples and lists, and quoted strings survive the field as one candidate apiece."""
    options = [ConfigValueRenderer.render(value) for value in DICT_VALUES] + TYPED

    assert _split([", ".join(options)], tmp_path) == [options]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the front-end literal helper")
def test_scalar_candidates_split_at_every_comma_and_drop_blanks(tmp_path):
    """Plain scalar candidates split at each comma, with surrounding blanks and empty entries removed."""
    assert _split(["50, 200 ,, 400", "", "cuda"], tmp_path) == [["50", "200", "400"], [], ["cuda"]]
