#!/usr/bin/env python
"""Report what the available artefacts support for Arabic generation (Issue #15).

Thin wrapper around ``ansg.generation.readiness``; torch-free. See
``docs/GENERATION_READINESS.md`` for the levels and their evidence requirements.
"""
import sys

from ansg.generation.readiness import main

if __name__ == "__main__":
    sys.exit(main())
