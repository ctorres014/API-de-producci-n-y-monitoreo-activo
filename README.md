# Orquestador Multi-Agente — API de Producción (Entrega 7)

FastAPI asíncrona sobre el orquestador LangGraph del Módulo 6, con Redis para
persistencia de estado, LangSmith/Phoenix para observabilidad y un nodo
Human-in-the-Loop (HITL) que pausa la ejecución ante tareas críticas.

> El CLI original del Módulo 6 (`python main.py`) sigue funcionando sin cambios.

---

## Arquitectura

```
POST /tasks ──► worker (BackgroundTask en threadpool)
                    │
                    ▼
             LangGraph graph  ──► RedisSaver (checkpoints)
           ┌──────────────────────────────────────────────┐
           │  START → supervisor → research → supervisor  │
           │                    ↘ human_approval          │
           │                         │ (approved)         │
           │                    → analyst → supervisor    │
           │                    ↘ END                     │
           └──────────────────────────────────────────────┘
                    │
                    ▼
              Redis (job state)   ──► LangSmith (trazas)
```

| Componente | Responsabilidad |
|------------|----------------|
| `app/main.py` | FastAPI: endpoints asíncronos, sin bloquear el event loop |
| `app/graph.py` | Grafo extendido con HITL + RedisSaver checkpointer |
| `app/worker.py` | Ejecución en background, transiciones de estado en Redis |
| `app/observability.py` | Init LangSmith o Arize Phoenix |
| `app/hitl.py` | Nodo `human_approval` con `interrupt()` de LangGraph |

---

## Prerrequisitos

- Python 3.12+
- Redis 7+ (local o Docker)

---

## Inicio rápido (sin Docker)

### 1. Clonar e instalar

```bash
git clone https://github.com/ctorres014/Orquestador-multi-agente-especializado.git
cd Orquestador-multi-agente-especializado
git checkout entrega7

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Variables de entorno

```bash
cp .env.example .env
```

Editá `.env` con tus claves reales (mínimo indispensable):

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

REDIS_URL=redis://localhost:6379

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=entrega7-multi-agent
```

### 3. Levantar Redis

```bash
# Con Docker (recomendado):
docker run -d --name redis -p 6379:6379 redis:7-alpine

# O con Redis instalado localmente:
redis-server
```

### 4. Iniciar la API

```bash
uvicorn app.main:app --reload
```

- API: `http://localhost:8000`
- Docs interactivos: `http://localhost:8000/docs`

---

## Inicio rápido (Docker Compose)

```bash
cp .env.example .env   # editá con tus claves
docker compose up --build
```

Levanta Redis y la API juntos.

---

## Uso de la API

### Enviar una tarea

```bash
curl -s -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}' | jq
```

Respuesta inmediata (sin bloquear):
```json
{
  "job_id": "3f8a2c1d-...",
  "status": "pending",
  "message": "Task enqueued. Poll GET /tasks/{job_id} for status."
}
```

### Consultar estado

```bash
curl -s http://localhost:8000/tasks/3f8a2c1d-... | jq
```

| Estado | Significado |
|--------|-------------|
| `pending` | Encolado, aún no iniciado |
| `running` | El grafo está ejecutándose |
| `awaiting_approval` | Pausado en nodo HITL, esperando aprobación |
| `done` | Completado exitosamente |
| `rejected` | El humano rechazó la tarea en HITL |
| `failed` | Error → estado actualizado en Redis (no loop infinito) |

### Aprobar / rechazar una tarea crítica (HITL)

Cuando la consulta contiene keywords como `deploy`, `delete`, `execute`,
`publish`, `send email`, etc., el grafo se pausa automáticamente y el
status cambia a `awaiting_approval`.

```bash
# Aprobar → continúa al agente analista
curl -s -X POST http://localhost:8000/tasks/3f8a2c1d-.../approve \
  -H "Content-Type: application/json" \
  -d '{"approved": true}' | jq

# Rechazar → finaliza el grafo
curl -s -X POST http://localhost:8000/tasks/3f8a2c1d-.../approve \
  -H "Content-Type: application/json" \
  -d '{"approved": false}' | jq
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## Prueba de carga — 5 peticiones concurrentes

Con la API corriendo:

```bash
python load_test.py
```

El script envía 5 consultas en paralelo, hace polling hasta que terminan,
calcula la latencia p95 y guarda los resultados en `load_test_results.json`.

**Capturar métricas del dashboard:**
1. Abrí `https://smith.langchain.com` → proyecto **entrega7-multi-agent**
2. En la pestaña **Runs** encontrás las 5 trazas
3. Capturá: vista general, traza individual, costo por ejecución y latencia p95
4. Guardá las capturas en `/screenshots`

---

## Observabilidad

### LangSmith (recomendado — cero instalación extra)

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=entrega7-multi-agent
```

LangChain auto-instrumenta todas las llamadas al LLM y cada nodo del grafo.

### Arize Phoenix (alternativa)

```bash
pip install arize-phoenix openinference-instrumentation-langchain
```

```env
PHOENIX_ENABLED=true
```

Dashboard en `http://localhost:6006`.

---

## Estructura del proyecto

```
├── app/
│   ├── main.py            # FastAPI: POST /tasks · GET /tasks/{id} · POST /tasks/{id}/approve
│   ├── graph.py           # Grafo con HITL routing + RedisSaver checkpointer
│   ├── worker.py          # Ejecución en background, estados en Redis
│   ├── observability.py   # Init LangSmith / Arize Phoenix
│   └── hitl.py            # Nodo human_approval: interrupt() + routing post-aprobación
├── agents/
│   ├── supervisor.py      # Agente supervisor (M6)
│   ├── research_agent.py  # Agente de investigación (M6)
│   └── analyst_agent.py   # Agente analista (M6)
├── tools/
│   ├── search_tools.py    # web_search, wikipedia_search
│   └── analysis_tools.py  # sentiment_analyzer, math_calculator, ...
├── state.py               # OrchestratorState (+ campos is_critical, is_approved)
├── config.py              # Factory de LLM (anthropic / openai / ollama)
├── main.py                # CLI del M6 (sin cambios)
├── load_test.py           # Script de carga: 5 peticiones concurrentes
├── docker-compose.yml     # API + Redis
├── Dockerfile
├── requirements.txt
├── .env.example
└── screenshots/           # Capturas del dashboard de observabilidad
```

---

## Variables de entorno — referencia completa

| Variable | Default | Descripción |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | Backend LLM: `anthropic`, `openai`, `ollama` |
| `ANTHROPIC_API_KEY` | — | Clave de Anthropic |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Modelo de Anthropic |
| `OPENAI_API_KEY` | — | Clave de OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | Modelo de OpenAI |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL del servidor Ollama |
| `OLLAMA_MODEL` | `llama3.2` | Modelo de Ollama |
| `TAVILY_API_KEY` | — | Clave de Tavily Search (opcional) |
| `REDIS_URL` | `redis://localhost:6379` | URL de conexión a Redis |
| `JOB_TTL_SECONDS` | `86400` | TTL de jobs en Redis (segundos) |
| `LANGCHAIN_TRACING_V2` | `false` | Habilitar trazas LangSmith |
| `LANGCHAIN_API_KEY` | — | Clave de LangSmith |
| `LANGCHAIN_PROJECT` | `entrega7-multi-agent` | Proyecto en LangSmith |
| `PHOENIX_ENABLED` | `false` | Habilitar Arize Phoenix |
| `MAX_ITERATIONS` | `6` | Iteraciones máximas del supervisor |

---

## Checklist de entrega

- [x] Endpoint asíncrono (`POST /tasks`) que encola sin bloquear el event loop
- [x] Estado del job en Redis: `PENDING → RUNNING → DONE / FAILED`
- [x] Excepciones capturadas → status `FAILED` (cliente no queda en loop infinito)
- [x] `RedisSaver` para checkpoints de LangGraph (fallback a `MemorySaver`)
- [x] LangSmith / Arize Phoenix configurado en `observability.py`
- [x] Nodo `human_approval` con `interrupt()` para tareas críticas
- [x] `POST /tasks/{id}/approve` para reanudar el grafo tras aprobación
- [x] `load_test.py` para las 5 peticiones concurrentes
- [x] `docker-compose.yml` con API + Redis
- [x] `screenshots/` con guía para capturar métricas del dashboard

---

## CLI del Módulo 6 (sin cambios)

El orquestador original sigue disponible:

```bash
python main.py
python main.py --query "What is quantum computing?"
python main.py --query "Analyze AI regulation" --verbose
```

---

## Licencia

MIT
