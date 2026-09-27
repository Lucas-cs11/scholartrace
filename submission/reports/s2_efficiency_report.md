# S2 Efficiency Evaluation Report

**Corpus**: pasa_realscholar_test_b3b570411ce2399c (22 queries, 184 Gold)

## FAST

### Offline Replay Cost (Actual)
- Physical HTTP calls: 0 (0.0/query)
- LLM calls: 74 total (3.4/query, median=3.0, P90=5.8)
- Tokens: 174466 total (7930.3/query, median=7081.5)
- Latency: mean=14112.9ms, median=12577.4ms, P90=24205.8ms

### Production-Equivalent Logical Cost
- Logical API calls/query: NOT_MEASURED
- Estimated production latency: NOT_MEASURED

**Note**: Offline replay uses cached responses (physical_http=0). Production deployment would require real OpenAlex API calls. Logical cost cannot be reconstructed from frozen artifacts without SearchTrace.

### Candidate Volume
- Mean retrieval candidates: 100.5
- Mean final output: 10.6

## DEEP

### Offline Replay Cost (Actual)
- Physical HTTP calls: 0 (0.0/query)
- LLM calls: 109 total (5.0/query, median=5.0, P90=6.7)
- Tokens: 260343 total (11833.8/query, median=11807.0)
- Latency: mean=21953.0ms, median=21982.8ms, P90=30522.3ms

### Production-Equivalent Logical Cost
- Logical API calls/query: NOT_MEASURED
- Estimated production latency: NOT_MEASURED

**Note**: Offline replay uses cached responses (physical_http=0). Production deployment would require real OpenAlex API calls. Logical cost cannot be reconstructed from frozen artifacts without SearchTrace.

### Candidate Volume
- Mean retrieval candidates: 100.5
- Mean final output: 13.0

---

## Summary

- **Offline validation**: Both FAST and DEEP successfully replay with 0 API calls (cache hit 100%)
- **LLM cost**: DEEP requires ~1.5x LLM calls vs FAST due to Round-2 reranking
- **Production cost**: Cannot measure without SearchTrace containing logical API call counts
- **Efficiency tradeoff**: FAST is default mode (lower cost); DEEP provides optional deep search (higher cost, limited F1 gain)

**S2-B Status**: ✅ EFFICIENCY_REPORT = PASS