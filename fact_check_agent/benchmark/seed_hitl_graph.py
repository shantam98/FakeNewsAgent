"""Seed Neo4j with speaker-topic credibility edges from the LIAR training set.

Run once before benchmarking to populate the HITL credibility graph.
The resulting graph has Source nodes (speakers) with base_credibility derived
from their PolitiFact credit history, ready for the router's HITL override layer.

Usage:
    python -m fact_check_agent.benchmark.seed_hitl_graph \\
        --train-path /path/to/liar/train.tsv
"""
import argparse
import logging

from fact_check_agent.src.memory_client import close_memory, get_memory
from fact_check_agent.benchmark.record import BenchmarkRecord, load_liar_dataset

logger = logging.getLogger(__name__)


def seed_speaker_credibility(records: list[BenchmarkRecord]) -> int:
    """Merge speaker nodes into Neo4j graph with credibility scores.

    Returns the number of speakers seeded.
    """
    memory = get_memory()
    seeded = 0

    for record in records:
        if record.dataset != "liar" or record.speaker_credibility is None:
            continue
        if not record.speaker:
            continue

        try:
            memory._graph.merge_source(
                source_id        = f"spk_{record.source_name.lower().replace(' ', '_')}",
                name             = record.source_name,
                domain           = record.source_domain,
                category         = f"speaker_{record.party_affiliation or 'unknown'}",
                base_credibility = record.speaker_credibility,
            )
            seeded += 1
        except Exception as e:
            logger.warning("Failed to seed speaker %s: %s", record.speaker, e)

    logger.info("Seeded %d speaker credibility nodes into Neo4j", seeded)
    return seeded


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Seed Neo4j with LIAR speaker credibility")
    parser.add_argument("--train-path", required=True, help="Path to LIAR train.tsv")
    args = parser.parse_args()

    records = load_liar_dataset(args.train_path, split="train")
    n = seed_speaker_credibility(records)
    print(f"Seeded {n} speakers into Neo4j.")
    close_memory()


if __name__ == "__main__":
    main()
