# Contributing to TraceAnnotate

Thank you for your interest in contributing to **TraceAnnotate**!

We welcome contributions of all kinds: bug fixes, new extractors, heuristic improvements, documentation enhancements, and additional embodied AI adapters.

---

## Code of Conduct & Core Philosophy

TraceAnnotate was forged from real-world, high-stakes production environments processing multi-terabyte raw media libraries. When contributing, keep our four core principles in mind:

1. **Zero-Trust Cold Storage Safety (R1-R5)**: Never write directly to raw source media drives. Staging must always be chunked, checksum-verified, and purged immediately.
2. **Field-Level Provenance**: Every predicted value must have a clear `fill` origin tag (`auto_metric`, `auto_rule`, `auto_path`, `candidate_only`, `pending_review`, `human_confirmed`). Never silently synthesize fake ground truth.
3. **Decouple Confirmed from Candidate**: Dirty field data contains edge cases (statues with faces, street signs with phone numbers). Always use composite heuristics (F1-F3) to tier annotations before marking them confirmed.
4. **Offline First**: All core processing must be capable of running fully offline without external cloud API dependencies.

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/trace-annotate.git
   cd trace-annotate
   ```

2. **Install in editable mode**:
   ```bash
   pip install -e ".[full,dev]"
   ```

3. **Run the test suite**:
   ```bash
   pytest tests/ -v
   ```

---

## Submitting Pull Requests

1. Create a feature branch (`git checkout -b feature/my-new-feature`).
2. Add comprehensive unit tests in `tests/` covering your changes.
3. Verify that all 33+ tests pass and no regression occurs.
4. Submit your pull request with a clear description of the problem solved and test evidence.
