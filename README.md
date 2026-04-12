---
title: Personaidentify Environment Server
emoji: 📺
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# PersonaBench

An OpenEnv benchmark for evaluating agents on shopper persona inference, product ranking, and cold-start preference elicitation. Three tasks of increasing difficulty test whether an agent can move from passive behavioral inference to active, efficient recommendation.

**Live environment:** `https://premsagars-personaidentify.hf.space/`  
**Web UI:** `https://premsagars-personaidentify.hf.space/web`  
**API docs:** `https://premsagars-personaidentify.hf.space/docs`

---

## Why This Benchmark

Real recommendation systems face three compounding problems: inferring who a user is from their purchase history, ranking plausible products under ambiguity, and handling cold-start users with no history at all. PersonaBench turns this progression into a concrete, scorable evaluation problem with typed interfaces, deterministic graders, and a shaped reward signal.

---

## Tasks

### Task 1 — Persona Inference

The agent receives a user's full purchase history and the persona catalogue. It returns persona predictions with confidence scores in `[0, 1]`. No follow-up questions are allowed.

**Grader:** confidence-weighted cosine similarity × magnitude ratio against the annotated persona distribution. Rewards calibrated predictions over raw label guessing.

### Task 2 — Product Ranking

The agent receives persona labels and a mixed basket of target products and same-category decoys. It returns a ranked list of product titles.

Same-category decoys make this genuinely hard — coarse category filtering is insufficient. The agent must use persona signals and basket context to discriminate within a category.

**Grader:** mean average precision (MAP) against the user's actual target purchases.

### Task 3 — Cold-Start Interrogation

No purchase history is provided. The agent questions an evaluator-backed shopper simulation under a budget of 5 questions, then submits persona predictions and a ranked product list.

**Grader:**

```
0.2 × persona_reward + 0.5 × ranking_reward + 0.3 × ((5 − questions_asked) / 5)
```

This rewards efficient, targeted elicitation. Exhausting the question budget is penalized.

---

## Scoring Summary

| Task | Grader | Output range |
|------|--------|-------------|
| T1 | Confidence-weighted cosine similarity | 0.0 – 1.0 |
| T2 | MAP vs. target purchases | 0.0 – 1.0 |
| T3 | Weighted composite (T1 + T2 + efficiency) | 0.0 – 1.0 |

---

## Dataset

- `server/user_personas.json` — 146 user records with purchase history, persona annotations, and aggregate stats
- `server/persona_catalogue.json` — 10 persona definitions, each with a name, description, and behavioral signals

Each purchase history item contains `title`, `rating`, `price`, `description`, and `review_text`. Each user record includes a primary persona and a confidence-weighted `all` list of persona signals, enabling richer evaluation than single-label classification. Purchase histories range from 5 to 341 items.

---

## Action Schema

All actions are `PersonaIdentifyAction` objects with a `task` discriminator field.

**Task 1 — persona predictions:**

```json
{
  "task": 1,
  "predictions": [
    { "persona": "Collector", "confidence": 0.9 },
    { "persona": "Music Enthusiast", "confidence": 0.75 }
  ]
}
```

**Task 2 — ranked product list:**

```json
{
  "task": 2,
  "ranked_products": [
    "Dave's Picks Volume 11",
    "Tivoli Concert Hall, Copenhagen 4/14/1972",
    "Stars"
  ]
}
```

**Task 3 — question turn:**

```json
{
  "task": 3,
  "text_question": "What usually matters most to you when deciding whether to buy a music product?"
}
```

**Task 3 — final submission** (empty `text_question`, both fields set):

```json
{
  "task": 3,
  "predictions": [
    { "persona": "Collector", "confidence": 0.85 }
  ],
  "ranked_products": [
    "Limited Edition Live Recording",
    "Deluxe Anniversary Box Set"
  ]
}
```

**T3 state transition:** `text_question` set → questioning phase. `text_question` empty + `ranked_products` set → final submission, episode ends and reward is computed. If both are set, `ranked_products` takes priority and the episode is evaluated immediately.

`predictions[*].confidence` is validated to stay within `[0, 1]`.

---

## Observation Schema

All observations are `PersonaIdentifyObservation` objects. Common fields present in every observation:

| Field | Type | Description |
|-------|------|-------------|
| `task` | `1 \| 2 \| 3` | Active benchmark task |
| `reward` | `float` | Reward from the latest step |
| `done` | `bool` | Whether the episode has ended |
| `instruction` | `str` | Task-specific instruction text |

Task-specific fields:

| Field | Task | Description |
|-------|------|-------------|
| `purchase_history` | T1 | Full review-level purchase history for the current user |
| `personas` | T1, T2 | Persona catalogue entries exposed to the agent |
| `persona_labels` | T2 | Confidence-weighted persona labels for the current user |
| `basket` | T2 | Shuffled product basket (targets + same-category decoys) |
| `users_remaining` | T1, T2 | Users left in the current multi-user episode |
| `start_intro` | T3 | Opening message from the evaluator-backed shopper simulation |
| `text_reply` | T3 | Evaluator reply to the agent's latest question |

Episode state payload (returned alongside every observation):

```json
{
  "episode_id": "1c71dd8f-18b9-4a68-94c5-54f8d6f95fa5",
  "step_count": 3,
  "user_id": "AE2EQ2X45UHWINYAJFZXKS6NYJ2A",
  "task": 2
}
```

---

## Running Locally

### Task 3 evaluator setup

The shopper simulator in Task 3 is backed by `PersistentLLMHelper` in `llm.py`. Set these before running:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export LLM_MODEL_NAME=...
```

These can also go in a local `.env` file — `llm.py` calls `load_dotenv()` on startup.

### Start the server

```bash
uv sync
uv run uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Baseline inference script

`inference.py` runs a lightweight baseline over Tasks 1 and 2. Set:

```bash
export HF_TOKEN=...
export API_BASE_URL=...
export MODEL_NAME=...
```

Then run:

```bash
python3 inference.py
```

---

## Configuration

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `USERS_PER_EPISODE` | `server/personaidentify_environment.py` | `5` | Multi-user episode length for T1 and T2 |
| `MAXQ` | `evalhelpers.py` | `5` | T3 question budget |
| `W1, W2, W3` | `evalhelpers.py` | `0.2, 0.5, 0.3` | T3 reward weights |

---

## Repository Map

| File | Purpose |
|------|---------|
| `server/personaidentify_environment.py` | Core environment logic, episode handling, T3 orchestration |
| `evalhelpers.py` | Reward functions: persona similarity, MAP, T3 composite |
| `models.py` | OpenEnv action, observation, and state models |
| `datamodels.py` | Typed product, review, persona, and prediction data models |
| `llm.py` | Persistent evaluator LLM helper for T3 shopper simulation |
| `utils.py` | Persona loading, purchase extraction, basket construction |
| `server/app.py` | FastAPI app factory for OpenEnv serving |
| `openenv.yaml` | OpenEnv runtime manifest |
| `inference.py` | Baseline driver for T1 and T2 evaluation |
| `validate-submission.sh` | Submission validation workflow |
