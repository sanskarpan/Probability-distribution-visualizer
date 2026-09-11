# Archive — historical development notes

These documents were written during early development and are preserved for
provenance. **They are stale by design** — counts, file trees, and status claims
in them predate the production hardening pass and should not be treated as
current documentation.

Current docs: `README.md`, `QUICKSTART.md`, `CHANGELOG.md`, and the MkDocs site
in `docs/` (published to GitHub Pages).

| File | What it was |
|---|---|
| `ADVANCED_FEATURES.md` | Early multivariate/copula/mixture manual (superseded by `docs/advanced.md`) |
| `AUDIT_LOG.md` | Round-by-round audit history |
| `FINAL_REPORT.md` | 23-phase production report (claims 557 tests / 82.4% — see real counts below) |
| `FIXES.md` | List of 54 fixes |
| `ISSUES.md` | Early issue list (35 issues, 40% coverage baseline) |
| `PROJECT_SUMMARY.md` | Original spec (2117 LOC / 45+ tests era) |
| `REVIEW_PROMPT.md` | Oversized review mega-prompt used during development |
| `TEST_VERIFICATION_REPORT.md` | 2026-02-03 verification snapshot (56/56 claim — predates suite growth) |

Ground truth as of the hardening pass: **688 tests passing** (`pytest tests/ -q`),
`src/` + `web/` as mapped in `docs/architecture.md`.
