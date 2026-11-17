# Results

## End to end

| model | agent | accuracy | accuracy_table | accuracy_documents | routing_accuracy | llm_calls_per_q | tokens_per_q | seconds_per_q |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct | rag-only | 0.300 | 0.000 | 0.600 | 0.500 | 1.000 | 749.225 | 0.626 |
| qwen2.5-1.5b-instruct | router+specialists | 0.633 | 0.833 | 0.433 | 0.867 | 2.783 | 827.000 | 1.282 |
| qwen2.5-1.5b-instruct | single-agent | 0.525 | 0.600 | 0.450 | 0.908 | 2.075 | 871.067 | 1.187 |
| qwen2.5-3b-instruct | rag-only | 0.317 | 0.000 | 0.633 | 0.500 | 1.000 | 747.217 | 1.623 |
| qwen2.5-3b-instruct | router+specialists | 0.725 | 0.833 | 0.617 | 0.992 | 2.492 | 765.800 | 2.182 |
| qwen2.5-3b-instruct | single-agent | 0.717 | 0.833 | 0.600 | 1.000 | 2.050 | 897.733 | 3.173 |

## Table questions: SQL execution

| model | agent | sql_failed | sql_retried |
|---|---|---|---|
| qwen2.5-1.5b-instruct | rag-only | 0.000 | 0.000 |
| qwen2.5-1.5b-instruct | router+specialists | 0.033 | 0.050 |
| qwen2.5-1.5b-instruct | single-agent | 0.017 | 0.067 |
| qwen2.5-3b-instruct | rag-only | 0.000 | 0.000 |
| qwen2.5-3b-instruct | router+specialists | 0.000 | 0.000 |
| qwen2.5-3b-instruct | single-agent | 0.033 | 0.100 |

## Document questions: retrieval and answer overlap

| model | agent | doc_f1 | retrieval_hit |
|---|---|---|---|
| qwen2.5-1.5b-instruct | rag-only | 0.500 | 0.883 |
| qwen2.5-1.5b-instruct | router+specialists | 0.369 | 0.633 |
| qwen2.5-1.5b-instruct | single-agent | 0.389 | 0.650 |
| qwen2.5-3b-instruct | rag-only | 0.523 | 0.883 |
| qwen2.5-3b-instruct | router+specialists | 0.506 | 0.883 |
| qwen2.5-3b-instruct | single-agent | 0.486 | 0.850 |

## Accuracy by question template

| model | agent | count_region | free_city | over_region | piaf | top_region | total_city | total_region |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct | rag-only | 0.000 | 0.000 | 0.000 | 0.600 | 0.000 | 0.000 | 0.000 |
| qwen2.5-1.5b-instruct | router+specialists | 0.889 | 0.938 | 0.857 | 0.433 | 0.375 | 0.917 | 0.875 |
| qwen2.5-1.5b-instruct | single-agent | 0.889 | 0.000 | 1.000 | 0.450 | 0.375 | 0.917 | 0.875 |
| qwen2.5-3b-instruct | rag-only | 0.000 | 0.000 | 0.000 | 0.633 | 0.000 | 0.000 | 0.000 |
| qwen2.5-3b-instruct | router+specialists | 0.889 | 0.562 | 1.000 | 0.617 | 1.000 | 0.833 | 1.000 |
| qwen2.5-3b-instruct | single-agent | 0.889 | 0.562 | 0.857 | 0.600 | 1.000 | 1.000 | 0.875 |

## Retrieval on all 3,835 PIAF questions (761 paragraphs)

| retriever | questions | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 |
|---|---|---|---|---|---|---|
| bm25 | 3835 | 0.671 | 0.802 | 0.839 | 0.881 | 0.744 |
| dense | 3835 | 0.668 | 0.832 | 0.872 | 0.909 | 0.757 |
| hybrid | 3835 | 0.706 | 0.851 | 0.890 | 0.920 | 0.784 |
