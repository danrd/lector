"""Lets `python -m lector` work, alongside the `lector` console script."""
from lector.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
