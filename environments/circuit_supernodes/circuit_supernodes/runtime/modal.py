"""Modal-only access to subject-model services."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed

retry_modal_call = retry(
    stop=stop_after_attempt(4),
    wait=wait_chain(wait_fixed(1), wait_fixed(4), wait_fixed(15)),
    reraise=True,
)


class SubjectConfig(BaseModel):
    """Subject model and graph-export settings selected by a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = "Qwen/Qwen3-4B"
    transcoder_set: str = "mwhanna/qwen3-4b-transcoders"
    node_threshold: float = Field(0.8, gt=0.0, le=1.0)
    edge_threshold: float = Field(0.98, gt=0.0, le=1.0)


class ModalSubject:
    """Client for one deployed Modal subject-model class."""

    def __init__(
        self, app_name: str, class_name: str, config: SubjectConfig | None = None
    ) -> None:
        self.app_name = app_name
        self.class_name = class_name
        self.config = config or SubjectConfig()
        self._instance = None

    def _subject(self):
        if self._instance is None:
            import modal

            self._instance = modal.Cls.from_name(self.app_name, self.class_name)(
                model_name=self.config.model_name,
                transcoder_set=self.config.transcoder_set,
                node_threshold=self.config.node_threshold,
                edge_threshold=self.config.edge_threshold,
            )
        return self._instance

    def _method(self, name: str):
        method = getattr(self._subject(), name, None)
        if method is None:
            raise AttributeError(
                f"{self.app_name}/{self.class_name} has no method {name!r}"
            )
        return method

    async def call(self, method: str, *args: Any) -> Any:
        return await self._method(method).remote.aio(*args)

    def map(
        self,
        method: str,
        *inputs: Iterable[Any],
        order_outputs: bool = True,
    ) -> Iterable[Any]:
        return self._method(method).map(*inputs, order_outputs=order_outputs)

    def spawn(self, method: str, *args: Any) -> Any:
        return self._method(method).spawn(*args)

    def sample_answers(
        self, requests: Sequence[dict[str, object]]
    ) -> list[dict[str, object]]:
        return list(self.map("sample_until_answer", requests, order_outputs=True))

    def attribution_graphs(self, prompts: Sequence[str]) -> list[dict | None]:
        return list(self.map("attribute_prompt", prompts, order_outputs=True))

    def spawn_attribution_graph(self, prompt: str) -> Any:
        return self.spawn("attribute_prompt", prompt)


@lru_cache
def subject_from_config(config: SubjectConfig) -> ModalSubject:
    return ModalSubject("circuit-supernodes", "Attributor", config)


SUBJECT = subject_from_config(SubjectConfig())
