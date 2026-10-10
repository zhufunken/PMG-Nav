# Snapshot Validation

Date: 2026-10-10. Environment: macOS ARM64, Python 3.12, pytest 9.1.1,
NumPy 2.5.3, PyYAML 6.0.3.

Command: `python -m pytest tests/pmg_nav -q`.

Result: 111 passed, 9 failed. The original project and this source snapshot
produce identical failed test names in the same validation environment:

- `test_direct_stall_guard.py::RouteProgressGuardTests::test_invalid_route_clears_route_progress_latch`
- `test_pmg_nav.py::test_m3_parity_builds_m1_payload_without_action_authority`
- `test_pmg_nav.py::test_m1_is_stateless_and_m3_corrects_conflict`
- `test_pmg_nav.py::test_simple_oracle_stability_uses_its_dedicated_score_threshold`
- `test_simple_oracle_latch.py::test_candidate_and_route_latch_ignore_motion_and_reordering`
- `test_simple_oracle_latch.py::test_arrival_is_sticky_for_single_target`
- `test_simple_oracle_latch.py::test_terminal_entry_replans_once_without_changing_candidate`
- `test_simple_oracle_latch.py::test_terminal_scope_rechecked_after_far_candidate_selection`
- `test_subset_gate.py::test_subset_gate_requires_complete_paired_rows`

Python syntax compilation passed for `pmg_nav`, `scripts`, `autodl`, and `tests`.
No tracked file exceeds 10 MB. A scan for common GitHub, Hugging Face, and OpenAI
token patterns and PEM private-key headers found no matches; this is a limited
pattern scan, not a comprehensive security audit.

Isaac Sim, GPU baseline inference, and end-to-end navigation experiments were
not run. Existing implementation and regression failures were preserved during
repository preparation. Markdown hard line breaks and existing whitespace were
also preserved.
