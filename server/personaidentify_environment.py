# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Personaidentify Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

import json
import random
import math

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

class PersonaidentifyEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = PersonaidentifyEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Personaidentify environment ready!"
        >>>
        >>> obs = env.step(PersonaidentifyAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        """Initialize the personaidentify environment."""
        self._state = PersonaIdentifyState(episode_id=str(uuid4()), step_count=0, user_id="", task=1)
        self._reset_count = 0

        self.DATA = json.load(open("server/user_personas.json"))
        self.PERSONADATA = json.load(open("server/persona_catalogue.json"))


    def reset(self, task: int = 2) -> PersonaIdentifyObservation:    
        """
        Reset the environment.

        Returns:
            PersonaidentifyObservation with a ready message
        """
        self.user = random.choice(self.DATA)
        self.user_id = self.user['user_id']
        self.task = task

        self._state = PersonaIdentifyState(episode_id=str(uuid4()), step_count=0, user_id=self.user_id, task=self.task)
        self._reset_count += 1

        if task == 2:
            return PersonaIdentifyObservation(
                task=2,
                done = False,
                reward=0.0,
                personas=get_all_personas(self.PERSONADATA),
                basket=make_basket(self.DATA, self.user_id),
                persona_labels=get_personas(self.DATA, self.user_id)
            )

        purchaseHistory = []
        for item in self.user['purchase_history']:
            purchaseHistory.append(ProductReview(title=item['title'], 
                                                 rating=item['rating'], 
                                                 price=item['price'],
                                                 description=item['description'],
                                                 review_text=item['review_text']
                                                 ))


        return PersonaIdentifyObservation(
            done = False,
            reward=0.0,
            task=1,
            purchase_history=purchaseHistory,
            personas=get_all_personas(self.PERSONADATA)
        )

    def step(self, action: PersonaIdentifyAction) -> PersonaIdentifyObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: PersonaidentifyAction containing the message to echo

        Returns:
            PersonaidentifyObservation with the echoed message and its length
        """
        self._state.step_count += 1

        assert self.task == action.task

        if self.task == 2:
            reward = calculate_product_ranking_reward(get_real_purchases(self.DATA, self.user_id), action.ranked_products)
            return PersonaIdentifyObservation(
                task=2,
                done=True,
                reward=reward,
            )

        reward = calculate_persona_reward(self.user["persona"]["all"], action.predictions)
        return PersonaIdentifyObservation(
            task=1,
            done=True,
            reward=reward,
        )

    @property
    def state(self) -> PersonaIdentifyState:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
