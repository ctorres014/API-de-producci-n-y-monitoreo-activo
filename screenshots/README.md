# Screenshots — Observability Evidence

Place your dashboard captures here before submitting.

## Required captures

| File | What to capture |
|------|----------------|
| `phoenix_traces_overview.png` | Phoenix Traces tab showing the 5 concurrent runs |
| `phoenix_single_trace.png` | One expanded trace showing all graph nodes (supervisor → research → analyst) |
| `phoenix_latency_tokens.png` | Latency and token usage per execution |

## How to take them

1. Make sure the API is running with `PHOENIX_ENABLED=true` in `.env`.
2. Run `python3 load_test.py` to generate 5 concurrent traces.
3. Open `http://localhost:6006` → Phoenix dashboard.
4. In the **Traces** tab you will see the 5 traces — screenshot the list view.
5. Click into any single trace to see the node-by-node breakdown — screenshot the waterfall.
6. Use the latency / token columns for cost & latency evidence.
7. Save captures to this `screenshots/` folder.

## LangSmith alternative

If using LangSmith instead (`LANGCHAIN_TRACING_V2=true`), open `https://smith.langchain.com`
→ project **entrega7-multi-agent** and capture the equivalent views from the Runs tab.
