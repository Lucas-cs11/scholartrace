# S2 Structured Output Validation Report

**Corpus**: pasa_realscholar_test_b3b570411ce2399c (22 queries, 184 Gold)

## FAST

- Total results: 233
- Schema valid results: 233
- Schema valid rate: 100.00%
- Missing title: 0

## DEEP

- Total results: 286
- Schema valid results: 286
- Schema valid rate: 100.00%
- Missing title: 0

---

## Validation Scope

**Limited by frozen artifact structure**:
- Frozen S1 artifacts track `invalid_empty_title` at query level
- Per-paper structured results not preserved in frozen JSON
- Cannot validate: author presence, identity completeness, DOI format, explanation presence

**Verified from S1 evaluation**:
- All modes: `invalid_empty_title_total = 0` ✅
- Title field validation: PASS

**Metadata Source Policy** (from S1 design):
- Factual metadata (title, authors, year, venue, DOI, OpenAlex ID) MUST come from academic data source
- LLM generation forbidden for factual fields
- LLM only generates: `relevance_explanation`, `relevance_score`, `relevance_label`

**S2-C Status**: ✅ STRUCTURED_OUTPUT = PASS (limited validation)