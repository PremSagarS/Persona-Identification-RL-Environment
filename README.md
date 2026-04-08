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

# Persona Identification RL Environment

Persona Identification RL Environment is an OpenEnv benchmark for training and evaluating agents that reason over shopper behavior. The environment combines real purchase-history records, structured persona annotations, ranking tasks, and an evaluator-backed cold-start interaction setting. It is designed to measure whether an agent can move from passive behavioral inference to active preference elicitation and recommendation under a clean `reset` and `step` interface.

This repository packages the benchmark logic, reward functions, deployment surface, and a lightweight baseline driver in a single OpenEnv-compatible project. The result is a practical environment for studying sequential reasoning, recommendation quality, and efficient information gathering.

## Why This Benchmark Matters

Modern recommendation systems rarely start with a perfect user profile. In practice, systems must infer preferences from observed behavior when history exists, rank plausible purchases under ambiguity, and handle cold-start interactions when no history is available. This environment turns that progression into a concrete evaluation problem.

The benchmark emphasizes:

- real-world utility through persona-aware recommendation tasks grounded in purchase data,
- clear grader behavior through deterministic scoring for persona inference and product ranking,
- strong environment design through explicit episode boundaries and typed action and observation schemas,
- deployment readiness through OpenEnv, Docker, Hugging Face Spaces, and validation tooling,
- novelty through evaluator-backed cold-start interrogation rather than static one-shot prompting alone.

## Benchmark Design

The environment is organized as a three-stage difficulty ramp.

### Task 1: Persona Inference

The agent receives a user's full purchase history together with the persona catalogue. It returns one or more persona predictions, each with a confidence score in `[0, 1]`. This task measures whether the agent can translate observed shopping behavior into a structured shopper profile without asking any follow-up questions.

### Task 2: Product Ranking

The agent receives persona labels together with a basket that mixes target purchases and decoy products. It returns a ranked list of product titles. This task measures recommendation quality under noise, where the agent must use persona signals and basket context to surface the products most aligned with the user.

### Task 3: Cold-Start Interrogation

The agent starts without purchase history. Instead, it interacts with an evaluator-backed shopper simulation, asking natural-language questions until it is ready to submit persona predictions and a ranked product list. This task measures efficient information gathering, preference elicitation, and recommendation quality under a question budget.

Conceptually, the benchmark moves from passive inference to ranking and then to interactive cold start. In the implementation, Tasks 1 and 2 run as short multi-user episodes, while Task 3 runs as a single-user dialogue episode.

## Scoring and Evaluation

The benchmark is designed to reward both correctness and efficient interaction.

- Task 1 scores persona predictions against the user's annotated persona distribution using a normalized, confidence-weighted similarity objective.
- Task 2 scores ranked product titles with mean average precision (MAP) against the user's target purchases.
- Task 3 combines persona quality, recommendation quality, and question efficiency into a single reward.

The current Task 3 reward configuration is:

```text
0.2 * persona_reward + 0.5 * ranking_reward + 0.3 * ((MAXQ - questions_asked) / MAXQ)
```

with `MAXQ = 5`.

This grading structure gives the benchmark a useful mix of deterministic evaluation and interactive reasoning pressure:

- persona inference rewards calibrated structure rather than raw label guessing,
- recommendation quality is measured directly on ranked outputs,
- cold-start behavior is encouraged to be both informative and concise.

## Environment Mechanics

The environment follows the standard OpenEnv lifecycle:

1. `reset(task=...)` initializes a task-specific episode and returns the first observation.
2. `step(action)` scores the action for the current state and returns the next observation, reward, and `done` flag.
3. `state` exposes episode metadata for the active session.

Episode structure is task dependent:

- Tasks 1 and 2 sample `USERS_PER_EPISODE = 5` distinct users for each episode.
- Each `step(...)` call scores the current user and advances to the next one.
- The fifth user closes the episode with `done = true`.
- Task 3 samples a single user, generates a short shopper introduction, accepts up to five question turns, and ends when the agent submits its final persona and product predictions.

For Task 3, the environment seeds an evaluator LLM with the selected user's persona annotations and purchase history. The evaluator is instructed to stay in character, answer naturally, and avoid directly revealing persona labels or simply listing purchases back to the agent. This creates a controlled but still realistic cold-start interaction loop.

## Dataset and Data Construction

The benchmark is backed by two JSON assets:

- `server/user_personas.json`: 146 user records with `user_id`, persona annotations, aggregate stats, and purchase history.
- `server/persona_catalogue.json`: 10 persona definitions, each with a name, description, and behavioral signals.

Each purchase-history item contains review-level product data:

- `title`
- `rating`
- `price`
- `description`
- `review_text`

Each user record includes both a primary persona and an `all` list of confidence-weighted persona signals. This structure supports richer evaluation than single-label classification and lets the environment reward calibrated persona predictions.

The data supports a meaningful range of behavioral complexity:

- purchase histories span from 5 to 341 items,
- user-level records include aggregate purchase statistics,
- persona annotations capture both dominant and secondary traits.

Task 2 uses a concrete basket-construction policy implemented in `utils.py`:

- the ranking target is built from a fixed slice of 5 real purchases,
- up to 15 distinct decoy products are sampled from other users,
- the combined basket is shuffled before being returned to the agent.

This creates a 20-item ranking problem that favors preference reasoning over simple memorization.

## Action, Observation, and State Interfaces

The environment surface is defined in `models.py` and `datamodels.py`.

### Action Schema

Task 1 persona prediction:

```json
{
  "task": 1,
  "predictions": [
    {
      "persona": "Collector",
      "confidence": 0.9
    },
    {
      "persona": "Music Enthusiast",
      "confidence": 0.75
    }
  ]
}
```

Task 2 ranked recommendation:

```json
{
  "task": 2,
  "ranked_products": [
    "Dave's Picks Volume 11: Century II Convention Hall, Wichita, KS 11/17/72",
    "Tivoli Concert Hall, Copenhagen, Denmark 4/14/1972",
    "Stars"
  ]
}
```

Task 3 question turn:

```json
{
  "task": 3,
  "text_question": "What usually matters most to you when deciding whether to buy a music product?"
}
```

Task 3 final submission:

```json
{
  "task": 3,
  "predictions": [
    {
      "persona": "Collector",
      "confidence": 0.85
    },
    {
      "persona": "Music Enthusiast",
      "confidence": 0.72
    }
  ],
  "ranked_products": [
    "Limited Edition Live Recording",
    "Deluxe Anniversary Box Set",
    "Rare Import Vinyl"
  ]
}
```

`predictions[*].confidence` is validated to stay within `[0, 1]`.

### Observation Fields

Common fields:

- `task`: active benchmark task (`1`, `2`, or `3`)
- `reward`: reward returned for the latest step
- `done`: whether the episode has ended

Task-specific fields:

- `purchase_history`: full review-level history for Task 1
- `personas`: persona catalogue entries exposed to the agent
- `basket`: shuffled product basket for Task 2
- `persona_labels`: confidence-weighted persona labels for Task 2
- `users_remaining`: number of users left in the current Task 1 or Task 2 episode
- `start_intro`: evaluator-generated opening message for Task 3
- `text_reply`: evaluator reply for Task 3 question turns

Representative state payload:

```json
{
  "episode_id": "1c71dd8f-18b9-4a68-94c5-54f8d6f95fa5",
  "step_count": 3,
  "user_id": "AE2EQ2X45UHWINYAJFZXKS6NYJ2A",
  "task": 2
}
```

The state model contains:

- `episode_id`
- `step_count`
- `user_id`
- `task`

## Running the Environment

### Local Development

Install dependencies and start the FastAPI server:

```bash
uv sync
uv run uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Once the server is running:

- Web interface: `http://localhost:8000/web`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Task 3 uses an evaluator-backed shopper simulator. Configure the evaluator model before running interactive cold-start episodes:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export LLM_MODEL_NAME=...
```

The repository loads environment variables through `.env` when available.

### Docker

Build and run the container from the repository root:

```bash
docker build -t personaidentify-env .
docker run --rm -p 8000:8000 personaidentify-env
```

### Hugging Face Spaces Deployment

Deploy the environment with the OpenEnv CLI:

```bash
openenv push
```

The OpenEnv manifest is defined in `openenv.yaml` and points to `server.app:app` on port `8000`.

### Submission Validation

Validate the deployment surface with:

```bash
./validate-submission.sh <ping_url>
```

The validation script checks:

- live `/reset` reachability for the deployed Space,
- Docker build success,
- `openenv validate` compliance.

## Baseline Inference Script

`inference.py` is a lightweight baseline driver that exercises the benchmark by running one Task 1 episode and one Task 2 episode directly against `PersonaidentifyEnvironment`. It uses a chat model to generate persona predictions and ranked product outputs, then logs step-level rewards and episode summaries.

Configure the script with:

```bash
export HF_TOKEN=...
export API_BASE_URL=...
export MODEL_NAME=...
```

Optional runtime knobs:

- `MAX_TOKENS`
- `TEMPERATURE`
- `MAX_STEPS`
- `SUCCESS_SCORE_THRESHOLD`

Run the script with:

```bash
python3 inference.py
```

This makes it a convenient starting point for validating prompt strategies, model behavior, and reward sensitivity on the benchmark's non-dialogue tasks.

## Repository Map

- `server/personaidentify_environment.py`: core environment logic, task flow, episode handling, and Task 3 evaluator orchestration
- `evalhelpers.py`: reward functions for persona similarity, MAP scoring, and Task 3 combination logic
- `utils.py`: persona loading, purchase extraction, and basket construction helpers
- `models.py`: OpenEnv action, observation, and state models
- `datamodels.py`: typed product, review, persona, and prediction data models
- `llm.py`: persistent evaluator LLM helper used by the Task 3 shopper simulation
- `server/app.py`: FastAPI app factory wiring for OpenEnv serving
- `openenv.yaml`: OpenEnv runtime manifest for local validation and deployment
- `inference.py`: baseline driver for model-based Task 1 and Task 2 evaluation
- `validate-submission.sh`: submission validation workflow for deployment readiness

## Benchmark Configuration and Tuning

The benchmark exposes a compact set of configuration levers that shape task difficulty and reward balance:

- `USERS_PER_EPISODE = 5` in `server/personaidentify_environment.py` controls the multi-user episode length for Tasks 1 and 2.
- `MAXQ = 5` in `evalhelpers.py` sets the Task 3 question budget.
- `W1 = 0.2`, `W2 = 0.5`, and `W3 = 0.3` weight persona quality, ranking quality, and question efficiency in Task 3.
- basket construction in `utils.py` determines how many real purchases and sampled decoys appear in Task 2.
- evaluator model settings in `llm.py` shape the interaction behavior of the cold-start shopper simulation.

These controls make it straightforward to tune ranking difficulty, interaction cost, and reward composition while preserving the benchmark's overall structure.

## Summary

Persona Identification RL Environment brings together three capabilities that matter in real recommendation systems: inferring user identity from behavior, ranking likely products under uncertainty, and conducting efficient cold-start conversations. With typed models, explicit reward design, OpenEnv packaging, and deployable serving infrastructure, the repository provides a strong foundation for benchmarking agents on persona-aware recommendation workflows.
