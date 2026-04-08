# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Personaidentify Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PersonaidentifyAction, PersonaidentifyObservation, Task1Action, Task1Observation, Task1State


class PersonaidentifyEnv(
    EnvClient[PersonaidentifyAction, PersonaidentifyObservation, State]
):
    """
    Client for the Personaidentify Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with PersonaidentifyEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(PersonaidentifyAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = PersonaidentifyEnv.from_docker_image("personaidentify-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(PersonaidentifyAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: PersonaidentifyAction) -> Dict:
        """
        Convert PersonaidentifyAction to JSON payload for step message.

        Args:
            action: PersonaidentifyAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PersonaidentifyObservation]:
        """
        Parse server response into StepResult[PersonaidentifyObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with PersonaidentifyObservation
        """
        obs_data = payload.get("observation", {})
        observation = PersonaidentifyObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return Task1State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            user_id=payload.get("user_id", "ERROR")
        )
