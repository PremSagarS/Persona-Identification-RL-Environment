# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Personaidentify Environment.

The personaidentify environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, BaseModel, field_validator
from typing import Literal, List

import json

from datamodels import ProductReview, PersonaPrediction

class Task1Observation(Observation):
    task: Literal[1] = 1
    purchase_history: list[ProductReview]

class Task1Action(Action):
    predictions: list[PersonaPrediction]

    @field_validator('predictions', mode='before')
    @classmethod
    def handle_web_interface_strings(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # If it's not valid JSON, let Pydantic raise the standard error
                pass
        return v

class Task1State(State):
    user_id: str