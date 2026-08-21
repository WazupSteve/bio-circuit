"""Modal-only access to subject-model services."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed

retry_modal_call = retry(
    stop=stop_after_attempt(4),
    wait=wait_chain(wait_fixed(1), wait_fixed(4), wait_fixed(15)),
    reraise=True,
)


class ModalSubject:
    """Client for one deployed Modal subject-model class."""

    def __init__(self, app_name: str, class_name: str) -> None:
        self.app_name = app_name
        self.class_name = class_name
        self._instance = None

    def _subject(self):
        if self._instance is None:
            import modal

            self._instance = modal.Cls.from_name(self.app_name, self.class_name)()
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

    def call_sync(self, method: str, *args: Any) -> Any:
        return self._method(method).remote(*args)

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
