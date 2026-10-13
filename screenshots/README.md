# Screenshots — Observability Evidence

Place your dashboard captures here before submitting.

## Required captures

| File | What to capture |
|------|----------------|
| `langsmith_traces_overview.png` | LangSmith project view showing the 5 concurrent runs |
| `langsmith_single_trace.png` | One expanded trace showing all graph nodes (supervisor → research → analyst) |
| `langsmith_cost_per_run.png` | Token cost per execution (input + output tokens, cost in USD) |
| `langsmith_p95_latency.png` | p95 latency from the dashboard after the load test |

## How to take them

1. Run `python load_test.py` (API must be running with `LANGCHAIN_TRACING_V2=true`).
2. Open https://smith.langchain.com → project **entrega7-multi-agent**.
3. In the **Runs** tab you will see 5 traces — screenshot the list view.
4. Click into any single trace to see the node-by-node breakdown — screenshot the waterfall.
5. Use the **Monitor** tab (or the run list's token column) for cost & latency data.

## Phoenix alternative

If using Arize Phoenix (`PHOENIX_ENABLED=true`), open http://localhost:6006 and
capture the equivalent views from the Phoenix UI.
