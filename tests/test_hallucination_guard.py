"""
tests/test_hallucination_guard.py
----------------------------------
Unit tests for the hallucination guardrail.

Uses a small synthetic clause set so no parsed nodes JSON file is required.
Monkeypatches _load_known_clauses to inject a controlled set of known clauses.
"""

from __future__ import annotations

import pytest


KNOWN_CLAUSES = {
    "4.4.1.1", "4.4.1.2", "4.4.1.3",
    "6.2.2", "6.2.3",
    "3.1.2", "3.1.6",
    "2.4.2.4",
    "9.2.1.1", "9.2.2",
}


@pytest.fixture(autouse=True)
def patch_known_clauses(monkeypatch):
    """Replace the file-based clause loader with a known synthetic set."""
    import structrag.evaluation.hallucination_guard as hg
    monkeypatch.setattr(
        hg,
        "_load_known_clauses",
        lambda source_id="ec2_2004_nf": KNOWN_CLAUSES,
    )
    # Also reset the module-level cache so each test starts clean
    monkeypatch.setattr(hg, "_known_clauses", None)


class TestExtractCitedClauses:
    def test_explicit_clause_keyword(self):
        from structrag.evaluation.hallucination_guard import extract_cited_clauses
        result = extract_cited_clauses("See clause 6.2.2 for details.")
        assert "6.2.2" in result

    def test_section_symbol(self):
        from structrag.evaluation.hallucination_guard import extract_cited_clauses
        result = extract_cited_clauses("Per EC2 §4.4.1.2, the cover is 25 mm.")
        assert "4.4.1.2" in result

    def test_multiple_clauses(self):
        from structrag.evaluation.hallucination_guard import extract_cited_clauses
        text = "Per clause 6.2.2 and clause 9.2.1.1, the minimum is defined."
        result = extract_cited_clauses(text)
        assert "6.2.2" in result
        assert "9.2.1.1" in result

    def test_no_clauses(self):
        from structrag.evaluation.hallucination_guard import extract_cited_clauses
        result = extract_cited_clauses("The beam should be properly designed.")
        assert result == []

    def test_table_reference(self):
        from structrag.evaluation.hallucination_guard import extract_cited_clauses
        result = extract_cited_clauses("See Table 4.4N for cover values.")
        # Table references use a different pattern — check it doesn't crash
        assert isinstance(result, list)


class TestCheckHallucination:
    def test_clean_answer_known_clause(self):
        from structrag.evaluation.hallucination_guard import check_hallucination
        result = check_hallucination(
            "The cover is defined in clause 4.4.1.2 of Eurocode 2."
        )
        assert result.is_clean is True
        assert "4.4.1.2" in result.valid
        assert result.hallucinated == []

    def test_hallucinated_clause_detected(self):
        from structrag.evaluation.hallucination_guard import check_hallucination
        result = check_hallucination(
            "Per EC2 clause 7.9.4.12, the cover must be at least 50 mm."
        )
        assert result.is_clean is False
        assert "7.9.4.12" in result.hallucinated

    def test_one_level_sub_clause_allowed(self):
        """6.2.2.1 should be valid if 6.2.2 is in the known set."""
        from structrag.evaluation.hallucination_guard import check_hallucination
        result = check_hallucination(
            "As stated in clause 6.2.2.1, the formula applies."
        )
        # 6.2.2.1 parent (6.2.2) IS known — should not be hallucinated
        assert "6.2.2.1" not in result.hallucinated

    def test_no_citations_is_clean(self):
        from structrag.evaluation.hallucination_guard import check_hallucination
        result = check_hallucination(
            "The concrete must be properly cured before loading."
        )
        assert result.is_clean is True
        assert result.cited == []

    def test_mixed_valid_and_hallucinated(self):
        from structrag.evaluation.hallucination_guard import check_hallucination
        text = (
            "Per clause 4.4.1.2 the cover is 25 mm. "
            "See also clause 99.99.99 for clarification."
        )
        result = check_hallucination(text)
        assert result.is_clean is False
        assert "4.4.1.2" in result.valid
        assert "99.99.99" in result.hallucinated

    def test_annotated_answer_contains_warning_tag(self):
        from structrag.evaluation.hallucination_guard import check_hallucination
        result = check_hallucination(
            "Per EC2 clause 88.88.88, do this."
        )
        assert "NOT IN CORPUS" in result.annotated_answer

    def test_summary_clean(self):
        from structrag.evaluation.hallucination_guard import check_hallucination
        result = check_hallucination("No citations here.")
        summary = result.summary()
        assert isinstance(summary, str) and len(summary) > 0

    def test_summary_hallucinated(self):
        from structrag.evaluation.hallucination_guard import check_hallucination
        result = check_hallucination("See clause 55.55.55 for details.")
        assert "WARNING" in result.summary()
