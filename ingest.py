"""Entry point: Run PDF ingestion pipeline."""
from src.core.ingest import run_ingestion
import sys

if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else None
    run_ingestion(pdf)
