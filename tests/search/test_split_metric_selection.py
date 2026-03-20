"""Regression tests for split-aware metric selection across native databases."""

from skydiscover.config import (
    AdaEvolveDatabaseConfig,
    BeamSearchDatabaseConfig,
    BestOfNDatabaseConfig,
    DatabaseConfig,
    GEPANativeDatabaseConfig,
    OpenEvolveNativeDatabaseConfig,
)
from skydiscover.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.search.base_database import Program
from skydiscover.search.beam_search.database import BeamSearchDatabase
from skydiscover.search.best_of_n.database import BestOfNDatabase
from skydiscover.search.gepa_native.database import GEPANativeDatabase
from skydiscover.search.openevolve_native.database import OpenEvolveNativeDatabase
from skydiscover.search.topk.database import TopKDatabase


def _make_program(program_id: str, combined_score: float, train_score: float, val_score: float):
    return Program(
        id=program_id,
        solution=f"def solve(): return {program_id!r}",
        metrics={
            "combined_score": combined_score,
            "train_combined_score": train_score,
            "val_combined_score": val_score,
            "complexity": 0.5 + combined_score / 10,
            "diversity": 0.6 + combined_score / 10,
        },
    )


def test_native_databases_use_unprefixed_selection_metric_for_best_program():
    databases = [
        TopKDatabase("topk", DatabaseConfig()),
        BestOfNDatabase("best_of_n", BestOfNDatabaseConfig(best_of_n=2)),
        BeamSearchDatabase(
            "beam_search",
            BeamSearchDatabaseConfig(beam_width=2, beam_selection_strategy="best"),
        ),
        GEPANativeDatabase(
            "gepa_native",
            GEPANativeDatabaseConfig(candidate_selection_strategy="best"),
        ),
        AdaEvolveDatabase(
            "adaevolve",
            AdaEvolveDatabaseConfig(use_paradigm_breakthrough=False),
        ),
        OpenEvolveNativeDatabase(
            "openevolve_native",
            OpenEvolveNativeDatabaseConfig(feature_dimensions=["complexity", "diversity"]),
        ),
    ]

    for database in databases:
        overfit = _make_program("overfit", combined_score=0.3, train_score=0.95, val_score=0.3)
        generalizing = _make_program(
            "generalizing",
            combined_score=0.8,
            train_score=0.55,
            val_score=0.8,
        )
        add_kwargs = {"iteration": 0}
        if isinstance(database, AdaEvolveDatabase):
            add_kwargs["target_island"] = 0
        database.add(overfit, **add_kwargs)
        add_kwargs = {"iteration": 1}
        if isinstance(database, AdaEvolveDatabase):
            add_kwargs["target_island"] = 0
        database.add(generalizing, **add_kwargs)
        assert database.get_best_program().id == "generalizing"


def test_gepa_native_ignores_split_prefixed_metrics_in_metric_fronts():
    database = GEPANativeDatabase(
        "gepa_native",
        GEPANativeDatabaseConfig(candidate_selection_strategy="best"),
    )
    database.add(_make_program("p1", combined_score=0.4, train_score=0.9, val_score=0.4), iteration=0)
    database.add(_make_program("p2", combined_score=0.7, train_score=0.6, val_score=0.7), iteration=1)

    assert "train_combined_score" not in database.metric_best
    assert "val_combined_score" not in database.metric_best
    assert database.metric_best["combined_score"][0] == "p2"
