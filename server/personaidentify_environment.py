# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Personaidentify Environment Implementation.

Tasks 1 and 2 run across 5 users per episode. After each user the agent
receives an intermediate observation (done=False, cumulative reward so far).
After the 5th user the episode ends (done=True, final mean reward).
"""

from uuid import uuid4

import json
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PersonaIdentifyAction, PersonaIdentifyObservation, PersonaIdentifyState
    from ..datamodels import ProductReview
    from ..evalhelpers import calculate_persona_reward, calculate_product_ranking_reward
    from ..utils import get_all_personas, get_personas, make_basket, get_real_purchases
except ImportError:
    from models import PersonaIdentifyAction, PersonaIdentifyObservation, PersonaIdentifyState
    from datamodels import ProductReview
    from evalhelpers import calculate_persona_reward, calculate_product_ranking_reward
    from utils import get_all_personas, get_personas, make_basket, get_real_purchases


USERS_PER_EPISODE = 5


class PersonaidentifyEnvironment(Environment):
    """
    Personaidentify environment.

    Tasks 1 & 2 now span USERS_PER_EPISODE (5) users per episode.
    Each step() call scores the agent's answer for the current user and
    advances to the next. The episode ends (done=True) after all users
    have been evaluated, reporting the mean reward across the episode.

    Task 3 (cold-start) is single-turn per user as before.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        self._state = PersonaIdentifyState(episode_id=str(uuid4()), step_count=0, user_id="", task=1)
        self._reset_count = 0

        self.DATA = json.load(open("server/user_personas.json"))
        self.PERSONADATA = json.load(open("server/persona_catalogue.json"))

        # Multi-user episode tracking
        self._user_queue: list[dict] = []
        self._current_user: dict = {}
        self._user_index: int = 0

        self.task: int = 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_user_queue(self) -> list[dict]:
        """Sample USERS_PER_EPISODE distinct users for one episode."""
        return random.sample(self.DATA, k=min(USERS_PER_EPISODE, len(self.DATA)))

    def _load_current_user(self) -> None:
        """Point internal state at the user at _user_index in the queue."""
        self._current_user = self._user_queue[self._user_index]
        self._state.user_id = self._current_user["user_id"]

    def _observation_for_task1(self, *, done: bool, reward: float) -> PersonaIdentifyObservation:
        """Build a Task-1 observation for the current user."""
        purchase_history = [
            ProductReview(
                title=item["title"],
                rating=item["rating"],
                price=item["price"],
                description=item["description"],
                review_text=item["review_text"],
            )
            for item in self._current_user["purchase_history"]
        ]
        return PersonaIdentifyObservation(
            done=done,
            reward=reward,
            task=1,
            purchase_history=purchase_history,
            personas=get_all_personas(self.PERSONADATA),
            # Carry-along metadata the agent may find useful
            users_remaining=USERS_PER_EPISODE - self._user_index - (1 if done else 0),
        )

    def _observation_for_task2(self, *, done: bool, reward: float) -> PersonaIdentifyObservation:
        """Build a Task-2 observation for the current user."""
        uid = self._current_user["user_id"]
        return PersonaIdentifyObservation(
            task=2,
            done=done,
            reward=reward,
            personas=get_all_personas(self.PERSONADATA),
            basket=make_basket(self.DATA, uid),
            persona_labels=get_personas(self.DATA, uid),
            users_remaining=USERS_PER_EPISODE - self._user_index - (1 if done else 0),
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, task: int = 2) -> PersonaIdentifyObservation:
        """
        Reset the environment and start a fresh episode with a new user queue.

        Returns the first observation in the episode (user 1 of USERS_PER_EPISODE).
        """
        self.task = task
        self._user_queue = self._sample_user_queue()
        self._user_index = 0
        self._reset_count += 1

        self._state = PersonaIdentifyState(
            episode_id=str(uuid4()),
            step_count=0,
            user_id="",
            task=self.task,
        )
        self._load_current_user()

        if task == 2:
            return self._observation_for_task2(done=False, reward=0.0)

        # Task 1
        return self._observation_for_task1(done=False, reward=0.0)

    def step(self, action: PersonaIdentifyAction) -> PersonaIdentifyObservation:
        """
        Score the agent's answer for the current user, then advance to the
        next user (or end the episode if all users have been evaluated).

        Returns:
            - done=False + next user's observation  (users 1–4)
            - done=True  + final mean reward         (after user 5)
        """
        assert self.task == action.task, (
            f"Action task {action.task} does not match episode task {self.task}"
        )

        self._state.step_count += 1
        uid = self._current_user["user_id"]

        # ---- Score current user ----------------------------------------
        if self.task == 2:
            step_reward = calculate_product_ranking_reward(
                get_real_purchases(self.DATA, uid), action.ranked_products
            )
        else:  # task == 1
            step_reward = calculate_persona_reward(
                self._current_user["persona"]["all"], action.predictions
            )

        self._user_index += 1

        episode_done = self._user_index >= USERS_PER_EPISODE

        # ---- Build return observation -----------------------------------
        if episode_done:
            # Return step_reward for the last user — trainer sums across all steps
            return PersonaIdentifyObservation(
                task=self.task,
                done=True,
                reward=step_reward,
                users_remaining=0,
            )

        # Advance to next user and return their observation
        self._load_current_user()

        if self.task == 2:
            return self._observation_for_task2(done=False, reward=step_reward)

        return self._observation_for_task1(done=False, reward=step_reward)

    @property
    def state(self) -> PersonaIdentifyState:
        return self._state