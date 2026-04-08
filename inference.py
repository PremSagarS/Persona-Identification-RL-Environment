"""
Inference Script
================
It runs one episode for `task=1` (persona predictions) and one episode for `task=2` (product ranking)
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from server.personaidentify_environment import PersonaidentifyEnvironment
from models import PersonaIdentifyAction
from datamodels import PersonaPrediction

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
BENCHMARK = "personaidentify"

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

SYSTEM_PROMPT_P1 = textwrap.dedent(
    """
    You will be given a user's purchase history and a list of candidate personas.
    Respond with a JSON array of persona predictions (objects with fields `persona` and `confidence`),
    where `confidence` is a float in [0, 1]. Return nothing else.
    Example: [{"persona": "BargainHunter", "confidence": 0.8}, {"persona": "TechEnthusiast", "confidence": 0.2}]
    """
).strip()

SYSTEM_PROMPT_P2 = textwrap.dedent(
    """
    You will be given a shopping `basket` containing product titles and descriptions.
    Return a JSON array of product titles in ranked order (most likely purchases first).
    Only return the JSON array — no commentary.
    Example: ["Product A", "Product C", "Product B"]
    """
).strip()


def log_start(task: int, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _call_model_for_text(client: OpenAI, system: str, user_prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return ""


def parse_json_or_fallback(text: str):
    try:
        return json.loads(text)
    except Exception:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return None


def get_persona_predictions(client: OpenAI, obs) -> List[PersonaPrediction]:
    user_prompt = json.dumps(
        {
            "purchase_history": [
                {"title": p.title, "rating": p.rating, "price": p.price, "description": p.description, "review_text": p.review_text}
                for p in (obs.purchase_history or [])
            ],
            "personas": [ {"name": p.name, "description": p.description} for p in (obs.personas or []) ],
        }
    )

    text = _call_model_for_text(client, SYSTEM_PROMPT_P1, user_prompt)
    parsed = parse_json_or_fallback(text)

    if isinstance(parsed, list):
        preds = []
        for item in parsed:
            try:
                persona = item.get("persona") if isinstance(item, dict) else None
                confidence = float(item.get("confidence", 1.0)) if isinstance(item, dict) else 1.0
                if persona:
                    preds.append(PersonaPrediction(persona=persona, confidence=confidence))
            except Exception:
                continue
        if preds:
            return preds

    if getattr(obs, "personas", None):
        take = min(2, len(obs.personas))
        confidences = [1.0 / take] * take if take > 0 else []
        return [PersonaPrediction(persona=p.name, confidence=float(c)) for p, c in zip(obs.personas[:take], confidences)]

    return [PersonaPrediction(persona="unknown", confidence=1.0)]


def get_ranked_products(client: OpenAI, obs) -> List[str]:
    basket_serialized = [ {"title": p.title, "price": p.price, "description": p.description} for p in (obs.basket or []) ]
    user_prompt = json.dumps({"basket": basket_serialized})

    text = _call_model_for_text(client, SYSTEM_PROMPT_P2, user_prompt)
    parsed = parse_json_or_fallback(text)

    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
        return parsed

    return [p.title for p in (obs.basket or [])]


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = PersonaidentifyEnvironment()

    # Task 1 : persona predictions
    log_start(task=1, env=BENCHMARK, model=MODEL_NAME)
    try:
        obs = env.reset(task=1)

        preds = get_persona_predictions(client, obs)

        action = PersonaIdentifyAction(task=1, predictions=[ {"persona": p.persona, "confidence": p.confidence} for p in preds ])
        action_str = json.dumps([{"persona": p.persona, "confidence": p.confidence} for p in preds], separators=(",", ":"))

        result = env.step(action)
        reward = float(result.reward or 0.0)
        done = bool(result.done)
        error = None

        log_step(step=1, action=action_str, reward=reward, done=done, error=error)

        score = reward
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=1, score=score, rewards=[reward])
    except Exception as e:
        print(f"[DEBUG] task1 failed: {e}", flush=True)

    # Task 2 : product ranking
    log_start(task=2, env=BENCHMARK, model=MODEL_NAME)
    try:
        obs = env.reset(task=2)

        ranked = get_ranked_products(client, obs)

        action = PersonaIdentifyAction(task=2, ranked_products=ranked)
        action_str = json.dumps(ranked, separators=(",", ":"))

        result = env.step(action)
        reward = float(result.reward or 0.0)
        done = bool(result.done)
        error = None

        log_step(step=1, action=action_str, reward=reward, done=done, error=error)

        score = reward
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=1, score=score, rewards=[reward])
    except Exception as e:
        print(f"[DEBUG] task2 failed: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
