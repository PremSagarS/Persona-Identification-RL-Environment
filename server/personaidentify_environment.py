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
    from ..models import Task1State, Task1Action, Task1Observation
    from ..datamodels import ProductReview
except ImportError:
    from models import Task1Observation, Task1Action, Task1State
    from datamodels import ProductReview

def calculate_persona_reward(true_personas_list, pred_actions):
    """
    Calculates a magnitude-aware cosine similarity reward.
    
    Args:
        true_personas_list: List of dicts e.g. [{'persona': 'A', 'confidence': 0.8}]
        pred_actions: List of objects with .persona and .confidence attributes
    """
    # 1. Convert to dictionaries for easy lookup
    TruePersonas = {p['persona']: p['confidence'] for p in true_personas_list}
    PredPersonas = {p.persona: p.confidence for p in pred_actions}

    # 2. Identify all active personas (the union)
    active_keys = set(TruePersonas.keys()) | set(PredPersonas.keys())
    
    if not active_keys:
        return 1.0  # Perfect match on an empty profile

    # 3. Calculate Dot Product and Squared Magnitudes
    dot_product = 0.0
    true_mag_sq = 0.0
    pred_mag_sq = 0.0

    for p in active_keys:
        t = TruePersonas.get(p, 0.0)
        p_val = PredPersonas.get(p, 0.0)
        
        dot_product += t * p_val
        true_mag_sq += t ** 2
        pred_mag_sq += p_val ** 2

    # 4. Handle Zero Vectors (Zero Prediction Trap)
    if true_mag_sq == 0 or pred_mag_sq == 0:
        # If truth has data but pred is empty (or vice versa), reward is 0.
        return 1.0 if true_mag_sq == pred_mag_sq else 0.0

    # 5. Calculate Magnitudes
    true_mag = math.sqrt(true_mag_sq)
    pred_mag = math.sqrt(pred_mag_sq)

    # 6. Calculate Cosine Similarity (The 'Angle' / Direction)
    cosine_sim = dot_product / (true_mag * pred_mag)

    # 7. Calculate Magnitude Ratio (The 'Scale' / Completeness)
    # We use min/max so that over-predicting also lowers the reward.
    magnitude_ratio = min(true_mag, pred_mag) / max(true_mag, pred_mag)

    # 8. Final Combined Reward
    # In your 3-persona vs 1-persona test, this drops 0.655 down to ~0.43
    reward = cosine_sim * magnitude_ratio

    # Bound check for floating point safety
    return max(0.0, min(1.0, reward))

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
        self._state = Task1State(episode_id=str(uuid4()), step_count=0, user_id="")
        self._reset_count = 0

        self.DATA = json.load(open("server/user_personas.json"))


    def reset(self) -> Task1Observation:    
        """
        Reset the environment.

        Returns:
            PersonaidentifyObservation with a ready message
        """
        self.user = random.choice(self.DATA)
        self.user_id = self.user['user_id']
        self._state = Task1State(episode_id=str(uuid4()), step_count=0, user_id=self.user_id)
        self._reset_count += 1

        purchaseHistory = []
        for item in self.user['purchase_history']:
            purchaseHistory.append(ProductReview(title=item['title'], 
                                                 rating=item['rating'], 
                                                 price=item['price'],
                                                 description=item['description'],
                                                 review_text=item['review_text']
                                                 ))


        return Task1Observation(
            done = False,
            reward=0.0,
            task=1,
            purchase_history=purchaseHistory
        )

    def step(self, action: Task1Action) -> Task1Observation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: PersonaidentifyAction containing the message to echo

        Returns:
            PersonaidentifyObservation with the echoed message and its length
        """
        self._state.step_count += 1
        reward = calculate_persona_reward(self.user["persona"]["all"], action.predictions)

        return Task1Observation(
            done=True,
            reward=reward,
            task=1,
            purchase_history=[]
        )

    @property
    def state(self) -> Task1State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
