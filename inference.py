"""
Inference Script
================
It runs one episode for `task=1` (persona predictions), one episode for `task=2` (product ranking) and one episode for `task=3` (asking questions + recommendation)
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

SYSTEM_PROMPT_P3 = textwrap.dedent(
        """
        You are interacting with a simulated shopper. You will be given the shopper's
        short introduction and the list of candidate personas (name + description).
        You must respond with a single JSON object describing your next action.

        Two action types are allowed:
        1) Question: {"action": "question", "text": "<your question>"}
        2) Recommend: {"action": "recommend", "predictions": [{"persona": "<label>", "confidence": <0-1>}], "ranked_products": ["prod A", "prod B", ...]}

        Priorities:
            W1 = 0.2
            W2 = 0.5
            W3 = 0.3
            e1 = reward for finding persona
            e2 = reward for product ranking quality
            e3 = (Maximum number of questions - number of questions asked) / Maximum number of questions

            Evaluation = e1 * W1 + e2 * W2 + e3 * W3

            According to the evaluation function, the highest priority is to find the recommended products accurately, then to minimize the number of questions asked.

        Rules:
        - Follow the priorities and evaluation function to decide whether to ask a question or make a recommendation.
        - If you need more information, ask a short clarifying question using the Question action.
        - If you are ready to recommend, use the Recommend action and include a ranked list of product titles
            and persona predictions with confidences that sum roughly to 1.0 (or leave confidences as reasonable values).
        - Return only the JSON object, no commentary.
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


def build_task3_prompt(start_intro: str, personas: list, history: List[dict]) -> str:
    personas_serialized = [ {"name": p.name, "description": p.description} for p in (personas or []) ]
    prompt = json.dumps(
        {
            "start_intro": start_intro,
            "personas": personas_serialized,
            "history": history,
        }
    )
    return prompt


def get_task3_action(client: OpenAI, start_intro: str, personas: list, history: List[dict]) -> dict:
    user_prompt = build_task3_prompt(start_intro, personas, history)
    text = _call_model_for_text(client, SYSTEM_PROMPT_P3, user_prompt)
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = parse_json_or_fallback(text)

    if isinstance(parsed, dict) and parsed.get("action") in ("question", "recommend"):
        return parsed

    # Fallback: ask a clarification question if we don't understand model output
    return {"action": "question", "text": "Can you tell me briefly what kinds of products you prefer?"}


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = PersonaidentifyEnvironment()

    # Task 1 : persona predictions
    log_start(task=1, env=BENCHMARK, model=MODEL_NAME)
    try:
        obs = env.reset(task=1)

        rewards: List[float] = []
        steps_taken = 0

        # Loop over users until environment signals done
        while True:
            preds = get_persona_predictions(client, obs)

            action = PersonaIdentifyAction(task=1, predictions=[{"persona": p.persona, "confidence": p.confidence} for p in preds])
            action_str = json.dumps([{"persona": p.persona, "confidence": p.confidence} for p in preds], separators=(",", ":"))

            result = env.step(action)
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = None

            steps_taken += 1
            rewards.append(reward)

            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

            obs = result

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    except Exception as e:
        print(f"[DEBUG] task1 failed: {e}", flush=True)

    # Task 2 : product ranking
    log_start(task=2, env=BENCHMARK, model=MODEL_NAME)
    try:
        obs = env.reset(task=2)

        rewards: List[float] = []
        steps_taken = 0

        while True:
            ranked = get_ranked_products(client, obs)

            action = PersonaIdentifyAction(task=2, ranked_products=ranked)
            action_str = json.dumps(ranked, separators=(",", ":"))

            result = env.step(action)
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = None

            steps_taken += 1
            rewards.append(reward)

            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

            obs = result

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    except Exception as e:
        print(f"[DEBUG] task2 failed: {e}", flush=True)

    # Task 3 : cold-start dialogue + recommendation
    log_start(task=3, env=BENCHMARK, model=MODEL_NAME)
    try:
        obs = env.reset(task=3)

        start_intro = getattr(obs, "start_intro", "")
        personas = getattr(obs, "personas", [])

        history: List[dict] = []
        rewards: List[float] = []

        # initial system-provided intro (agent should see and may ask)
        # We treat the initial intro as a text_reply from the user for history
        history.append({"from": "user", "text": start_intro})

        score = 0.0
        success = False

        for step in range(1, MAX_STEPS + 1):
            action_obj = get_task3_action(client, start_intro, personas, history)

            if action_obj.get("action") == "question":
                qtext = action_obj.get("text", "")
                action = PersonaIdentifyAction(task=3, text_question=qtext)

                result = env.step(action)
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                error = None

                # record the reply from the environment/evaluator
                reply = getattr(result, "text_reply", None) or getattr(result.observation, "text_reply", None) or ""
                history.append({"from": "agent", "text": qtext})
                history.append({"from": "user", "text": reply})

                rewards.append(reward)
                log_step(step=step, action=qtext, reward=reward, done=done, error=error)

                # continue until model recommends or budget ends
                if done:
                    break

                continue

            # Recommend action
            if action_obj.get("action") == "recommend":
                preds = action_obj.get("predictions") or []
                ranked = action_obj.get("ranked_products") or []

                action = PersonaIdentifyAction(task=3, predictions=preds, ranked_products=ranked)

                result = env.step(action)
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                error = None

                rewards.append(reward)
                log_step(step=step, action=json.dumps(action_obj, separators=(",", ":")), reward=reward, done=done, error=error)

                score = reward
                success = score >= SUCCESS_SCORE_THRESHOLD
                log_end(success=success, steps=step, score=score, rewards=rewards)
                break

            # Unknown action -> ask a fallback question
            fallback_q = "Can you tell me what kinds of products you like most?"
            action = PersonaIdentifyAction(task=3, text_question=fallback_q)
            result = env.step(action)
            reply = getattr(result, "text_reply", None) or getattr(result.observation, "text_reply", None) or ""
            history.append({"from": "agent", "text": fallback_q})
            history.append({"from": "user", "text": reply})
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            log_step(step=step, action=fallback_q, reward=reward, done=result.done, error=None)

        else:
            # If loop completes without recommendation, force a final recommend call
            # Ask the model to produce a final recommendation
            final_action = get_task3_action(client, start_intro, personas, history)
            preds = final_action.get("predictions") or []
            ranked = final_action.get("ranked_products") or []
            action = PersonaIdentifyAction(task=3, predictions=preds, ranked_products=ranked)
            result = env.step(action)
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            score = reward
            success = score >= SUCCESS_SCORE_THRESHOLD
            log_step(step=MAX_STEPS + 1, action=json.dumps(final_action, separators=(",", ":")), reward=reward, done=bool(result.done), error=None)
            log_end(success=success, steps=MAX_STEPS + 1, score=score, rewards=rewards)

    except Exception as e:
        print(f"[DEBUG] task3 failed: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
