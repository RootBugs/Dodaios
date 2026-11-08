"""OpenCodeBrain — the ACP exchange OpenMausBot's ``opencode acp`` pipeline makes.

The subprocess is faked: this file feeds the brain the JSON-RPC frames a real
OpenCode server sends and reads the frames the brain writes back. No binary,
network or temp folder in the user's data directory is touched.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from dodaios.agent_runtimes.warm import WarmPool, WarmSpec
from dodaios.core.protocols import BrainMessage, BrainRequest
from dodaios.plugins.brain import opencode as ocmod
from dodaios.plugins.brain.opencode import OpenCodeBrain, _config_for

_FAKE_AGENT = Path(__file__).resolve().parents[3] / "fakes" / "fake_acp_agent.py"


class _NoWarmPool:
    """A pool that never has a process: the cold spawn path is exercised."""

    async def lease(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@pytest.fixture(autouse=True)
def _cold_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake-subprocess tests below exercise the cold turn explicitly.

    Warm reuse has its own tests at the end of this file; here the pool always
    reports "no process" so the original spawn path (and its fake) runs.
    """
    monkeypatch.setattr(ocmod, "_warm_pool", lambda: _NoWarmPool())


def _req(text: str = "Hello") -> BrainRequest:
    return BrainRequest(messages=(BrainMessage(role="user", content=text),))


def _frame(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj) + "\n").encode("utf-8")


def _model_option(value: str = "opencode/big-pickle") -> dict[str, Any]:
    return {
        "id": "model",
        "type": "select",
        "category": "model",
        "currentValue": "",
        "options": [{"value": value, "name": value}],
    }


def _initialize_result() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "agentInfo": {"name": "OpenCode", "version": "2.2.5"},
            "agentCapabilities": {"loadSession": True},
        },
    }


def _session_result(*, config_options: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "sessionId": "ses_1",
            "configOptions": config_options if config_options is not None else [_model_option()],
        },
    }


def _chunk(text: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": "ses_1",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            },
        },
    }


def _prompt_result(
    *,
    stop_reason: str = "end_turn",
    usage: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if error is not None:
        return {"jsonrpc": "2.0", "id": 3, "error": error}
    return {
        "jsonrpc": "2.0",
        "id": 3,
        "result": {"stopReason": stop_reason, "usage": usage or {}},
    }


class _FakeStdin:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        for line in data.decode("utf-8").splitlines():
            if line.strip():
                self.frames.append(json.loads(line))

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def methods(self) -> list[str]:
        return [str(frame.get("method") or frame.get("id")) for frame in self.frames]


class _FakeStdout:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        await asyncio.sleep(0)
        return self._lines.pop(0) if self._lines else b""


class _FakeStderr:
    async def readline(self) -> bytes:
        return b""


class _FakeProc:
    def __init__(self, lines: list[dict[str, Any]]) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout([_frame(line) for line in lines])
        self.stderr = _FakeStderr()
        self.pid = 4242
        self.returncode: int | None = None
        self.killed = False

    async def wait(self) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


class _Spawned:
    """The fake process plus the argv/env the brain launched it with."""

    def __init__(self, proc: _FakeProc) -> None:
        self.proc = proc
        self.argv: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}


def _install(monkeypatch: pytest.MonkeyPatch, lines: list[dict[str, Any]]) -> _Spawned:
    spawned = _Spawned(_FakeProc(lines))

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        if not spawned.argv:
            spawned.argv = args
            spawned.kwargs = kwargs
            return spawned.proc
        # Any later spawn is the Windows taskkill in ``_stop``.
        return _FakeProc([])

    monkeypatch.setattr(ocmod, "_resolve_binary", lambda: "opencode")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return spawned


@pytest.mark.asyncio
async def test_complete_streams_the_acp_answer_and_applies_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned = _install(
        monkeypatch,
        [
            _initialize_result(),
            _session_result(),
            {
                "jsonrpc": "2.0",
                "id": 10,
                "result": {"configOptions": []},
            },
            _chunk("Hello "),
            _chunk("there"),
            _prompt_result(usage={"inputTokens": 12, "outputTokens": 5, "cachedReadTokens": 2}),
        ],
    )

    brain = OpenCodeBrain(model="opencode/big-pickle")
    deltas = [delta async for delta in brain.complete(_req())]

    text = "".join(delta.content or "" for delta in deltas)
    assert text == "Hello there"
    assert any(delta.finish_reason == "stop" for delta in deltas)
    usage = next(delta.usage for delta in deltas if delta.usage)
    assert usage == {
        "input_tokens": 12,
        "output_tokens": 5,
        "cache_hit_tokens": 2,
    }

    # The same exchange OpenMausBot makes: initialize -> session/new ->
    # set_config_option(model) -> prompt, all over one ACP process.
    assert spawned.argv[-1] == "acp"
    methods = spawned.proc.stdin.methods()
    assert methods[:3] == ["initialize", "session/new", "session/set_config_option"]
    config = spawned.proc.stdin.frames[2]
    assert config["params"] == {
        "sessionId": "ses_1",
        "configId": "model",
        "value": "opencode/big-pickle",
    }
    prompt = spawned.proc.stdin.frames[3]
    assert prompt["method"] == "session/prompt"
    assert prompt["params"]["prompt"] == [{"type": "text", "text": brain._render_prompt(_req())}]
    # A throwaway cwd, never the caller's repository.
    assert "jarvis-opencode-brain-" in str(spawned.kwargs["cwd"])
    assert spawned.proc.stdin.closed is True


@pytest.mark.asyncio
async def test_config_option_the_session_does_not_offer_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned = _install(
        monkeypatch,
        [
            _initialize_result(),
            # Live OpenCode 2.2.5 offers ``model`` and ``mode`` only; a
            # session that offers neither of the brain's picks still prompts.
            _session_result(config_options=[{"id": "mode", "type": "select"}]),
            _chunk("ok"),
            _prompt_result(),
        ],
    )

    brain = OpenCodeBrain(model="opencode/big-pickle", structured_prompts=False)
    deltas = [delta async for delta in brain.complete(_req())]

    assert "".join(delta.content or "" for delta in deltas) == "ok"
    methods = spawned.proc.stdin.methods()
    assert "session/set_config_option" not in methods
    assert methods[1:3] == ["session/new", "session/prompt"]


@pytest.mark.asyncio
async def test_a_failed_prompt_raises_instead_of_returning_an_empty_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        [
            _initialize_result(),
            _session_result(config_options=[]),
            _prompt_result(error={"code": -32000, "message": "provider exploded"}),
        ],
    )

    brain = OpenCodeBrain()
    with pytest.raises(RuntimeError, match="provider exploded"):
        async for _ in brain.complete(_req()):
            pass


@pytest.mark.asyncio
async def test_an_answer_less_turn_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        [
            _initialize_result(),
            _session_result(config_options=[]),
            _prompt_result(),
        ],
    )

    brain = OpenCodeBrain()
    with pytest.raises(RuntimeError, match="no answer"):
        async for _ in brain.complete(_req()):
            pass


def test_config_for_sends_only_slug_models_and_the_effort() -> None:
    request = BrainRequest(
        messages=(BrainMessage(role="user", content="hi"),),
        reasoning_effort="high",
    )
    assert _config_for("opencode/big-pickle", request) == {
        "model": "opencode/big-pickle",
        "effort": "high",
        "variant": "high",
    }
    # A bare alias is not an ACP model id; leave the agent on its own default.
    assert _config_for("big-pickle", request) == {
        "effort": "high",
        "variant": "high",
    }
    assert _config_for("", BrainRequest(messages=())) == {}


def test_subscription_connected_probes_the_binary_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ocmod, "_probe_cache", None)
    monkeypatch.setattr(ocmod, "_resolve_binary", lambda: "opencode")
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = b"opencode/big-pickle\nopencode/glm-5.3-flash-free\n"

    def fake_run(argv: list[str], **_: Any) -> _Result:
        calls.append(argv)
        return _Result()

    monkeypatch.setattr(ocmod.subprocess, "run", fake_run)

    assert OpenCodeBrain.subscription_connected() is True
    assert OpenCodeBrain.subscription_connected() is True
    assert calls == [["opencode", "models"]]


def test_subscription_connected_is_false_without_the_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ocmod, "_probe_cache", None)
    monkeypatch.setattr(ocmod, "_resolve_binary", lambda: None)

    assert OpenCodeBrain.subscription_connected() is False


def test_subscription_connected_is_false_when_no_model_is_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ocmod, "_probe_cache", None)
    monkeypatch.setattr(ocmod, "_resolve_binary", lambda: "opencode")

    class _Result:
        returncode = 0
        stdout = b""

    monkeypatch.setattr(ocmod.subprocess, "run", lambda *a, **k: _Result())

    assert OpenCodeBrain.subscription_connected() is False


@pytest.mark.asyncio
async def test_complete_raises_when_the_binary_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ocmod, "_resolve_binary", lambda: None)
    brain = OpenCodeBrain()
    with pytest.raises(RuntimeError, match="not installed"):
        async for _ in brain.complete(_req()):
            pass


# --------------------------------------------------------------- warm process


class _LeaseStdout:
    """Scripted lines, then silence: the turn keeps waiting for its result."""

    def __init__(self, lines: list[dict[str, Any]]) -> None:
        self._lines = [_frame(line) for line in lines]
        self._silence = asyncio.Event()

    async def readline(self) -> bytes:
        if self._lines:
            await asyncio.sleep(0)
            return self._lines.pop(0)
        await self._silence.wait()
        return b""


class _FakeLease:
    """The pool's process as the brain sees it (``warm.Lease`` shape)."""

    def __init__(self, lines: list[dict[str, Any]], *, initialized: dict[str, Any]) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _LeaseStdout(lines)
        self.stderr = _FakeStderr()
        self.initialized = initialized
        self.open_session = ""
        self.turn: Any = None
        self.finished = False
        self.killed = False

    def finish(self) -> None:
        self.finished = True

    def kill(self) -> None:
        self.killed = True


@pytest.mark.asyncio
async def test_a_warm_lease_skips_the_handshake_and_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _FakeLease(
        [
            _session_result(config_options=[]),
            _chunk("warm "),
            _chunk("answer"),
            _prompt_result(),
        ],
        initialized=_initialize_result()["result"],
    )

    async def fake_lease(_binary: str) -> _FakeLease:
        return lease

    monkeypatch.setattr(ocmod, "_lease_warm", fake_lease)
    deltas = [delta async for delta in OpenCodeBrain().complete(_req())]

    assert "".join(delta.content or "" for delta in deltas) == "warm answer"
    # No ``initialize``: the pool's process answered it once for every turn.
    assert lease.stdin.methods()[:2] == ["session/new", "session/prompt"]
    # The turn recorded itself, so the pool can remember its session.
    assert lease.turn is not None
    # A healthy process is handed back, never killed.
    assert lease.finished is True
    assert lease.killed is False


@pytest.mark.asyncio
async def test_a_cancelled_warm_turn_cancels_the_prompt_and_ends_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _FakeLease(
        [_session_result(config_options=[])],  # the prompt is never answered
        initialized=_initialize_result()["result"],
    )

    async def fake_lease(_binary: str) -> _FakeLease:
        return lease

    monkeypatch.setattr(ocmod, "_lease_warm", fake_lease)
    brain = OpenCodeBrain()

    async def consume() -> None:
        async for _ in brain.complete(_req()):
            pass

    task = asyncio.create_task(consume())
    for _ in range(200):
        if "session/prompt" in lease.stdin.methods():
            break
        await asyncio.sleep(0.01)
    assert "session/prompt" in lease.stdin.methods()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert lease.stdin.methods()[-1] == "session/cancel"
    assert lease.killed is True
    assert lease.finished is False


@pytest.mark.asyncio
async def test_turns_share_one_process_and_open_a_fresh_session_each(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pool runs one process; every turn still starts at ``session/new``.

    The conversation lives in the prompt, so reusing the previous ACP session
    would mix two unrelated turns: the brain's ``SessionWant`` must ask for a
    fresh one each time (``resume=None``, never bound).
    """
    pool = WarmPool("opencode-brain")
    log = tmp_path / "frames.log"
    env = dict(os.environ)
    env["FAKE_ACP_LOG"] = str(log)
    spec = WarmSpec(
        argv=[sys.executable, str(_FAKE_AGENT)],
        env=env,
        cwd=tmp_path,
        signature="fake-agent",
    )
    monkeypatch.setattr(ocmod, "_resolve_binary", lambda: "opencode")
    monkeypatch.setattr(ocmod, "_warm_pool", lambda: pool)
    monkeypatch.setattr(ocmod, "_warm_spec", lambda *_args: spec)

    brain = OpenCodeBrain(model="opencode/glm-5.3-flash-free")
    try:
        first = [delta async for delta in brain.complete(_req("one"))]
        second = [delta async for delta in brain.complete(_req("two"))]
    finally:
        await pool.stop()

    assert "".join(delta.content or "" for delta in first) == (
        "echo: " + brain._render_prompt(_req("one"))[-80:]
    )
    assert "".join(delta.content or "" for delta in second) == (
        "echo: " + brain._render_prompt(_req("two"))[-80:]
    )
    methods = [
        str(json.loads(line).get("method"))
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert methods == [
        "initialize",
        "session/new",
        "session/prompt",
        "session/new",
        "session/prompt",
    ]


@pytest.mark.asyncio
async def test_a_rebuilt_event_loop_drops_the_stale_warm_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-process app restart builds a new loop; the old pool must go.

    Its process and tasks belong to the loop that no longer runs, so leasing
    it would wait forever on a future nobody resolves. The brain drops it
    (killing the child) and builds a fresh pool on the running loop.
    """

    class _StalePool:
        abandoned = False

        def abandon(self) -> None:
            self.abandoned = True

    stale = _StalePool()
    old_loop = asyncio.new_event_loop()
    try:
        monkeypatch.undo()  # the autouse fixture hides the real pool factory
        monkeypatch.setattr(ocmod, "_warm_pool_instance", stale)
        monkeypatch.setattr(ocmod, "_warm_loop", old_loop)
        pool = ocmod._warm_pool()
    finally:
        old_loop.close()

    assert stale.abandoned is True
    assert pool is not stale
    assert ocmod._warm_loop is asyncio.get_running_loop()
