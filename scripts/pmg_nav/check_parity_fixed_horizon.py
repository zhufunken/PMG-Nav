#!/usr/bin/env python3
"""Offline AST/logic check for the fixed-horizon parity contract.

This check deliberately does not import Isaac Sim, PyTorch, YAML, or pytest.
It validates the compatibility runner's opt-in switches and executes the
small environment-flag helpers in isolation, so it can run on a developer
machine before spending GPU time.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import textwrap
import types


PROJECT_MAIN = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_MAIN / "autodl/scripts/20_run_dynanav_isaac6_compat.py"
BASELINE_GENERATOR = PROJECT_MAIN / "autodl/scripts/08_run_dynanav_baseline.sh"
OFFICIAL_MATRIX = PROJECT_MAIN / "scripts/pmg_nav/05_run_official_matrix.py"


def _function_nodes(tree: ast.AST) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _load_flag_helpers(nodes: dict[str, ast.AST]) -> types.SimpleNamespace:
    """Execute only the env-flag helpers, without importing runner deps."""
    required = ("_env_flag", "parity_fixed_horizon_enabled", "parity_deterministic_enabled")
    source = "import os\n" + "\n\n".join(ast.unparse(nodes[name]) for name in required)
    namespace: dict[str, object] = {}
    exec(compile(source, str(RUNNER), "exec"), namespace, namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in required})


def _check_parity_generation_wrapper() -> None:
    """Check the generator's opt-in greedy TIC-VLA wrapper.

    The closed-loop launcher is a shell script which materialises a patched
    behavior module through a Python heredoc.  Parsing the whole shell file as
    Python would therefore be both fragile and misleading.  Instead, keep a
    small source-level contract here: parity mode must wrap the released
    ``generate_and_extract_kv_cache`` call, force greedy generation, and leave
    the sampling-only knobs out of the copied generation configuration.  The
    checks intentionally look for semantic markers rather than line numbers,
    so ordinary upstream edits do not invalidate the offline gate.
    """
    if not BASELINE_GENERATOR.is_file():
        raise AssertionError(f"missing baseline generator: {BASELINE_GENERATOR}")
    source = BASELINE_GENERATOR.read_text(encoding="utf-8")
    required_fragments = (
        "generate_and_extract_kv_cache",
        # It must be opt-in; legacy Stage 1 runs retain TIC-VLA's stochastic
        # sampling configuration.
        "STAGE2_PARITY_DETERMINISTIC",
        "PMG_PARITY_DETERMINISTIC",
        # Greedy decoding is represented by an explicit false assignment, not
        # merely a lower temperature (which would remain stochastic).
        "do_sample",
        "False",
        # Sampling controls must be removed/neutralised on the copied config.
        "temperature",
        "top_p",
        "top_k",
        "deepcopy",
        "to_dict",
        "setattr",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise AssertionError(
            "parity greedy-generation wrapper contract is incomplete: "
            + ", ".join(missing)
        )
    # Guard against an accidental unconditional behavior change.  A wrapper
    # that is installed for every run would invalidate the released baseline,
    # so require the parity switch to occur before the marker in the generated
    # transformation block.  (The launcher may contain unrelated mentions of
    # the marker in comments; the first occurrence is sufficient for this
    # conservative source check.)
    # Accept the current descriptive marker and the name used in an earlier
    # draft so this checker remains useful while a generated launcher is being
    # rolled forward across machines.
    marker_candidates = (
        "_stage2_parity_greedy_wrapped",
        "_stage2_det_generation_wrapped",
    )
    present_markers = [marker for marker in marker_candidates if marker in source]
    if not present_markers:
        raise AssertionError(
            "parity greedy-generation wrapper marker is missing (expected one of: "
            + ", ".join(marker_candidates)
            + ")"
        )
    marker_at = source.find(present_markers[0])
    parity_at = min(
        index for index in (
            source.find("STAGE2_PARITY_DETERMINISTIC"),
            source.find("PMG_PARITY_DETERMINISTIC"),
        )
        if index >= 0
    )
    if parity_at > marker_at:
        raise AssertionError(
            "greedy wrapper marker appears before its parity-only guard; "
            "legacy stochastic behavior could be changed"
        )

    _mock_parity_generation_wrapper(source)


def _check_strict_matrix_launcher() -> None:
    """Ensure official matrix shards cannot silently omit parity switches."""
    if not OFFICIAL_MATRIX.is_file():
        raise AssertionError(f"missing official matrix launcher: {OFFICIAL_MATRIX}")
    source = OFFICIAL_MATRIX.read_text(encoding="utf-8")
    required = (
        "--strict-parity",
        "STAGE2_STRICT_PARITY",
        "strict_parity",
        "PMG_PARITY_FIXED_HORIZON",
        "STAGE2_PARITY_FIXED_HORIZON",
        "PMG_PARITY_DETERMINISTIC",
        "STAGE2_PARITY_DETERMINISTIC",
        "STAGE2_STRICT_PARITY_ACTIVE",
        '"parity_fixed_horizon"',
        '"parity_deterministic"',
        '"_strict_parity"',
    )
    missing = [fragment for fragment in required if fragment not in source]
    if missing:
        raise AssertionError(
            "strict matrix launcher contract is incomplete: " + ", ".join(missing)
        )
    # The command-line option and env flag must resolve before base_env is
    # constructed, otherwise shards would inherit only a partial setting.
    resolve_at = source.find("strict_parity =")
    base_env_at = source.find("base_env =")
    if resolve_at < 0 or base_env_at < 0 or resolve_at > base_env_at:
        raise AssertionError(
            "strict_parity must be resolved before matrix base_env construction"
        )


def _eval_string_expression(node: ast.AST, values: dict[str, str]) -> str:
    """Evaluate the string-only expression used by the shell heredoc."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_string_expression(node.left, values) + _eval_string_expression(
            node.right, values
        )
    raise AssertionError(
        "unexpected non-string expression while extracting generated behavior: "
        + ast.dump(node, include_attributes=False)[:300]
    )


def _extract_greedy_function_source(source: str) -> str:
    """Extract the actual nested greedy wrapper from the behavior generator."""
    heredoc_starts = list(re.finditer(r"python3 - <<'PY'\n", source))
    for match in heredoc_starts:
        block_start = match.end()
        block_end = source.find("\nPY", block_start)
        if block_end < 0:
            continue
        block = source[block_start:block_end]
        if "inference_call =" not in block or "_stage2_greedy_generate" not in block:
            continue
        tree = ast.parse(block, filename=str(BASELINE_GENERATOR))
        values: dict[str, str] = {}
        replacement: str | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id == "inference_call":
                    values[target.id] = _eval_string_expression(node.value, values)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "inference_call"
                and len(node.args) >= 2
            ):
                replacement = _eval_string_expression(node.args[1], values)
                break
        if replacement is None:
            continue
        # The replacement starts with the existing command marker at a deeper
        # indentation.  Parse the generated prefix after its first uniform
        # eight-space line; the nested function itself is then recoverable via
        # AST regardless of comments or line wrapping.
        function_anchor = replacement.find("        # TIC-VLA's released")
        if function_anchor < 0:
            function_anchor = replacement.find("        # TIC-VLA")
        if function_anchor < 0:
            continue
        generated_prefix = replacement[function_anchor:]
        # The replacement expression appends the original ``inference_call``
        # marker after the injected diagnostics.  It intentionally ends in an
        # opening parenthesis, so trim that marker before parsing the prefix.
        original_call_marker = generated_prefix.rfind("\n        with torch.no_grad():")
        if original_call_marker >= 0:
            generated_prefix = generated_prefix[:original_call_marker]
        generated = textwrap.dedent(generated_prefix)
        generated_tree = ast.parse(generated, filename=str(BASELINE_GENERATOR))
        functions = [
            node
            for node in ast.walk(generated_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_stage2_greedy_generate"
        ]
        if functions:
            return ast.unparse(functions[0])
    raise AssertionError("could not extract _stage2_greedy_generate from baseline generator")


def _mock_parity_generation_wrapper(source: str) -> None:
    """Execute the generated wrapper against dict/object call signatures."""
    function_source = _extract_greedy_function_source(source)

    class MockGenerationConfig:
        def __init__(self) -> None:
            self.do_sample = True
            self.temperature = 0.1
            self.top_p = 0.1
            self.top_k = 10
            self.max_new_tokens = 8

        def to_dict(self) -> dict[str, object]:
            return {
                "do_sample": self.do_sample,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_new_tokens": self.max_new_tokens,
            }

    def make_wrapper():
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return "sentinel"

        factory_source = "def _factory(_stage2_original_generate):\n"
        factory_source += textwrap.indent(function_source, "    ")
        factory_source += "\n    return _stage2_greedy_generate\n"
        namespace: dict[str, object] = {}
        exec(compile(factory_source, str(BASELINE_GENERATOR), "exec"), namespace, namespace)
        return namespace["_factory"](original), calls

    def assert_config(value: object, *, original: object) -> None:
        if isinstance(value, dict):
            assert value.get("do_sample") is False
            assert all(key not in value for key in ("temperature", "top_p", "top_k"))
        else:
            assert getattr(value, "do_sample") is False
            assert getattr(value, "temperature", None) is None
            assert getattr(value, "top_p", None) is None
            assert getattr(value, "top_k", None) is None
            # The wrapper must not mutate the released config object.
            assert getattr(original, "do_sample") is True
            assert getattr(original, "temperature") == 0.1
            assert getattr(original, "top_p") == 0.1
            assert getattr(original, "top_k") == 10

    # Third positional argument: object and plain dict snapshots.
    wrapper, calls = make_wrapper()
    original_object = MockGenerationConfig()
    assert wrapper("img", "instruction", original_object) == "sentinel"
    assert len(calls) == 1
    assert_config(calls[-1][0][2], original=original_object)

    wrapper, calls = make_wrapper()
    original_dict = {
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.1,
        "top_k": 10,
        "max_new_tokens": 8,
    }
    assert wrapper("img", "instruction", original_dict) == "sentinel"
    assert len(calls) == 1
    assert_config(calls[-1][0][2], original=original_dict)
    assert original_dict["do_sample"] is True

    # Keyword generation_config: object preservation must work identically.
    wrapper, calls = make_wrapper()
    keyword_object = MockGenerationConfig()
    assert wrapper("img", "instruction", generation_config=keyword_object) == "sentinel"
    assert len(calls) == 1
    assert_config(calls[-1][1]["generation_config"], original=keyword_object)


def main() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER))
    nodes = _function_nodes(tree)
    _check_parity_generation_wrapper()
    _check_strict_matrix_launcher()
    required = {
        "_env_flag",
        "parity_fixed_horizon_enabled",
        "parity_deterministic_enabled",
        "configure_parity_determinism",
        "seed_episode",
        "run_navigation_with_manual_behavior",
    }
    missing = sorted(required - set(nodes))
    if missing:
        raise AssertionError(f"missing parity functions: {missing}")

    # Structural checks protect the actual early-break and seed call sites.
    for token in (
        "PMG_PARITY_FIXED_HORIZON",
        "STAGE2_PARITY_FIXED_HORIZON",
        "CUBLAS_WORKSPACE_CONFIG",
        "torch.use_deterministic_algorithms",
        "first_goal_reached_frame",
        "if not fixed_horizon:",
        "original_goal_probe",
        "runner._is_goal_reached = _parity_goal_probe",
    ):
        if token not in source:
            raise AssertionError(f"parity invariant missing from runner: {token}")

    helpers = _load_flag_helpers(nodes)
    tracked = ("PMG_PARITY_FIXED_HORIZON", "STAGE2_PARITY_FIXED_HORIZON", "PMG_PARITY_DETERMINISTIC", "STAGE2_PARITY_DETERMINISTIC")
    old = {name: os.environ.get(name) for name in tracked}
    try:
        for name in tracked:
            os.environ.pop(name, None)
        assert helpers.parity_fixed_horizon_enabled() is False
        assert helpers.parity_deterministic_enabled() is False

        os.environ["PMG_PARITY_FIXED_HORIZON"] = "1"
        assert helpers.parity_fixed_horizon_enabled() is True
        # Fixed-horizon parity also enables deterministic setup by default.
        assert helpers.parity_deterministic_enabled() is True

        os.environ.pop("PMG_PARITY_FIXED_HORIZON")
        os.environ["STAGE2_PARITY_FIXED_HORIZON"] = "yes"
        assert helpers.parity_fixed_horizon_enabled() is True

        os.environ.pop("STAGE2_PARITY_FIXED_HORIZON")
        os.environ["PMG_PARITY_DETERMINISTIC"] = "on"
        assert helpers.parity_fixed_horizon_enabled() is False
        assert helpers.parity_deterministic_enabled() is True
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    print(
        json.dumps(
            {
                "status": "passed",
                "runner": str(RUNNER),
                "baseline_generator": str(BASELINE_GENERATOR),
                "official_matrix_launcher": str(OFFICIAL_MATRIX),
                "checked": sorted(required),
                "mode": "offline_ast_and_flag_logic",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
