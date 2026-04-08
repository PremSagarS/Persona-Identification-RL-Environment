# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Personaidentify Environment.
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import field_validator
from typing import Literal, List

import json

from datamodels import ProductReview, PersonaPrediction, Product, Persona


class PersonaIdentifyObservation(Observation):
    task: Literal[1, 2]
    instruction: str = "TODO"
    # Task 1
    purchase_history: List[ProductReview] | None = None
    # Task 2
    persona_labels: List[PersonaPrediction] | None = None
    personas: List[Persona] | None = None
    basket: List[Product] | None = None


class PersonaIdentifyAction(Action):
    task: Literal[1, 2]
    # Task 1
    predictions: List[PersonaPrediction] | None = None
    # Task 2
    ranked_products: List[str] | None = None

    @field_validator('ranked_products', mode='before')
    @classmethod
    def handle_web_interface_strings_ranked_products(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        return v

    @field_validator('predictions', mode='before')
    @classmethod
    def handle_web_interface_strings_predictions(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        return v


class PersonaIdentifyState(State):
    task: Literal[1, 2]
    user_id: str