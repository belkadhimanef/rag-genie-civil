"""
tests/test_domain_classifier.py
---------------------------------
Unit tests for the domain classifier in groq_generation.py.

Tests verify that the classifier correctly allows EC2-related queries and
blocks clearly off-topic queries. No API keys or ChromaDB required.
"""

from __future__ import annotations

import pytest

from structrag.rag_pipeline.groq_generation import is_engineering_query


class TestEngineeringQueryClassifier:
    # ── Queries that must be allowed ──────────────────────────────────────

    def test_cover_query_fr(self):
        assert is_engineering_query("Quelle est la formule de l'enrobage nominal ?")

    def test_cover_query_en(self):
        assert is_engineering_query(
            "What is the minimum concrete cover for exposure class XC2?"
        )

    def test_safety_factor_fr(self):
        assert is_engineering_query(
            "Quelle est la valeur de γc en situation durable ?"
        )

    def test_safety_factor_en(self):
        assert is_engineering_query(
            "What partial safety factor applies to concrete in persistent situations?"
        )

    def test_shear_resistance(self):
        assert is_engineering_query(
            "Quelle est la résistance au cisaillement VRd,c sans armatures ?"
        )

    def test_exposure_class_list(self):
        assert is_engineering_query("Quelles sont les classes d'exposition XC ?")

    def test_concrete_class(self):
        assert is_engineering_query(
            "Quelle est la résistance fck pour le béton C30/37 ?"
        )

    def test_symbol_query(self):
        assert is_engineering_query("What is εcu2 for concrete C50/60?")

    def test_clause_number_reference(self):
        assert is_engineering_query("What does clause 4.4.1.2 say about cover?")

    def test_french_structural_term(self):
        assert is_engineering_query(
            "Calculer l'effort tranchant dans une poutre en béton armé"
        )

    def test_anchorage_question(self):
        assert is_engineering_query("Quelle est la longueur d'ancrage minimale ?")

    def test_english_reinforcement(self):
        assert is_engineering_query(
            "What is the minimum reinforcement ratio for a beam?"
        )

    # ── Queries that must be blocked ──────────────────────────────────────

    def test_weather_blocked(self):
        # The classifier uses "allow by default, block only clearly off-topic" logic.
        # "météo" (the French keyword) triggers the block; "temps" alone does not,
        # because "temps" is ambiguous (also means "time" in structural contexts).
        assert not is_engineering_query("Quelle est la météo à Paris demain ?")

    def test_cooking_blocked(self):
        assert not is_engineering_query(
            "Comment préparer une bonne recette de cuisine ?"
        )

    def test_finance_blocked(self):
        assert not is_engineering_query(
            "Quel est le cours du bitcoin aujourd'hui ?"
        )

    def test_eurocode_1_blocked(self):
        assert not is_engineering_query(
            "Quelle est la charge de vent selon l'Eurocode 1 ?"
        )

    def test_eurocode_8_blocked(self):
        assert not is_engineering_query(
            "Quel facteur de comportement q utiliser pour l'Eurocode 8 ?"
        )

    def test_seismic_blocked(self):
        assert not is_engineering_query(
            "What behavior factor should I use for seismic design per EN 1998?"
        )

    def test_medical_blocked(self):
        # "fracture" alone does not trigger the medical block (also a civil eng term).
        # "médecine" does trigger it.
        assert not is_engineering_query(
            "Quels sont les symptômes selon la médecine ?"
        )

    def test_greeting_blocked(self):
        assert not is_engineering_query("Bonjour, comment allez-vous ?")
