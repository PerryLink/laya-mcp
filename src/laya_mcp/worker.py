"""The model host: one warm model, serialised, with everything Laya leaves out.

This is the component that makes the difference between a script and a service.
It owns exactly one :class:`laya.Router` for the life of the process, and it
supplies the five things Laya does not:

**Warmth.** ``Router`` with ``preload`` keeps every checkpoint resident. The
alternative is Laya's default, ``max_loaded=1``, where alternating languages
rebuilds a model on every request - measured upstream at a 7.4 s median reload on
CPU and 10.3 s on a T4.

**Serialisation.** Laya is not thread-safe. ``Agent.system_one`` reassigns
``self.device`` and ``self.dtype`` and calls ``self.model.to(...)`` when it hits
an OOM, and it mutates ``self.cfg`` reads on every call. Concurrent calls can
therefore race a device move against a forward pass. A lock is not optional
politeness here; it is correctness. The default is one forward pass at a time.

**Preflight.** Every request is planned and validated before it reaches the
model, so a caller is told what would be truncated and which question is too
large, rather than receiving a confident answer about a fragment.

**Device honesty.** Laya catches a CUDA OOM and permanently demotes itself to
CPU in fp32, printing to stdout, with no flag set anywhere. This class watches
for that and reports it, so a client is never silently served answers an order of
magnitude slower than the ones it benchmarked.

**Structured failure.** A bare ``KeyError`` from Laya becomes a
:class:`~laya_mcp.errors.LayaMcpError` with a code, the offending question id,
and a hint.

**Lifecycle.** ``stop()`` releases the model and empties the CUDA cache, which
``Router.unload`` does not do. Issue #52 upstream is a long-running MLX sidecar
that grew to 21.7 GB; a service that cannot be recycled cleanly is a service that
eventually has to be killed.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .calibration import CalibrationEntry, CalibrationStore, apply_temperature
from .capability import (
    Capability,
    find_model_root,
    normalise_checkpoint,
    read_capability,
)
from .errors import (
    CapacityError,
    DeviceDegradedError,
    InvalidQuestionError,
    LayaMcpError,
    OutOfMemoryError,
    StateTruncatedError,
    UnknownModelError,
    translate,
)
from .planning import BudgetPlan, plan_questions, script_caveat
from .protocol import (
    Answer,
    AskRequest,
    AskResponse,
    Question,
    RoutingInfo,
    Usage,
)
from .validate import validate_questions

log = logging.getLogger("laya_mcp.worker")

#: Where a fitted calibration store lives by default, under the user's config dir.
DEFAULT_CALIBRATION_FILENAME = "calibration.json"

#: Where checkpoints are snapshotted when a local root is not supplied. Kept
#: beside the user's cache rather than inside the package, because a checkpoint is
#: around 650 MB and the package may be installed into a read-only prefix.
DEFAULT_MODEL_DIRNAME = "laya-mcp-models"


def _band(probability: float) -> str:
    """Place a ``noul`` probability in a band.

    The boundaries are the ones the ecosystem's own self-consistency guidance
    uses: ``no`` below 0.30, ``uncertain`` from 0.30 through 0.70 inclusive,
    ``yes`` above 0.70. Both boundaries belong to ``uncertain``, because a
    probability sitting exactly on a threshold is the case the band exists to
    describe as unclear.

    This is reported *alongside* the raw probability, never instead of it: a
    calibrated probability is not a decision, and 0.51 rendered as a settled
    ``true`` invites a caller to branch on a coin toss.
    """
    if probability < 0.30:
        return "no"
    if probability > 0.70:
        return "yes"
    return "uncertain"


@dataclass
class WorkerConfig:
    """How to host the model.

    Defaults are chosen to be safe rather than fast, and each one is a decision
    recorded here rather than buried in a call site.
    """

    model: str = "english"
    also: tuple[str, ...] = ()
    device: Optional[str] = None
    model_root: Optional[str] = None
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    calibration_path: Optional[str] = None
    #: Forward passes allowed at once. 1 is the only value Laya is safe with; the
    #: knob exists because a deployment with several GPUs and one process per GPU
    #: may legitimately want otherwise, and because it makes the constraint visible.
    concurrency: int = 1
    #: When true, a request whose state would be truncated is refused rather than
    #: answered. Off by default: answering and reporting is more useful than
    #: refusing, and the report is what makes the answer interpretable.
    strict: bool = False
    preload: bool = True


class LayaWorker:
    """Owns the model. One instance per process."""

    def __init__(self, config: Optional[WorkerConfig] = None) -> None:
        self.config = config or WorkerConfig()
        self._router: Any = None
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None
        self._capabilities: dict[str, Capability] = {}
        self._degraded = False
        self._requested_device = self.config.device
        self._calibration = CalibrationStore(self.config.calibration_path)
        self._calls = 0
        self._failures = 0
        self._started_at: Optional[float] = None
        self._last_latency_ms = 0.0

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Load the model. Idempotent.

        Raises :class:`~laya_mcp.errors.ModelUnavailableError` if it cannot, so a
        supervisor learns at startup rather than on the first request.
        """
        with self._state_lock:
            if self._loaded:
                return
            if self._load_error is not None:
                raise translate(RuntimeError(self._load_error))
            try:
                self._load()
            except LayaMcpError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                raise translate(exc)
            self._loaded = True
            self._started_at = time.time()

    def _load(self) -> None:
        import laya  # imported here so `doctor` never pays for torch

        wanted = [self.config.model, *self.config.also]
        normalised: list[str] = []
        for name in wanted:
            try:
                normalised.append(normalise_checkpoint(name))
            except KeyError as exc:
                raise UnknownModelError(
                    f"unknown checkpoint {name!r}",
                    hint="known checkpoints are english, multilingual and typed-decisions",
                    cause=exc,
                ) from exc

        # Deduplicate but keep order, so `--model english --also english` is not
        # an error and does not load twice.
        seen: list[str] = []
        for name in normalised:
            if name not in seen:
                seen.append(name)

        if self.config.model_root:
            # A local root bypasses the Hub entirely. Agent accepts either a repo
            # id or a directory, so passing the path is the whole change - but the
            # subfolder must then be resolved by us, because Agent joins it itself
            # and would double it.
            root = find_model_root(self.config.model_root)
            router = self._build_local_router(laya, root, seen)
        else:
            kwargs: dict[str, Any] = {}
            if self.config.device:
                kwargs["device"] = self.config.device
            router = laya.Router(preload=bool(self.config.preload), **kwargs)
            try:
                router.preload(seen)
            except Exception as exc:  # noqa: BLE001
                raise translate(exc) from exc

        self._router = router

        # Apply the token-budget overrides to every loaded agent. These are read
        # fresh on every `system_one` call, so setting them once at startup is
        # enough and setting them per request would be wasted work.
        for name in seen:
            try:
                agent = router.load(name)
            except Exception as exc:  # noqa: BLE001
                raise translate(exc) from exc
            self._apply_overrides(agent)
            self._capabilities[name] = self._describe(name, agent)

    def _build_local_router(self, laya_module: Any, root: str, names: Sequence[str]) -> Any:
        """A router whose checkpoints all come from one local directory."""
        import os

        models: dict[str, Any] = {}
        for name in names:
            if name == "english":
                models[name] = (root, None)
            else:
                subfolder = os.path.join(root, name)
                if not os.path.isdir(subfolder):
                    raise UnknownModelError(
                        f"checkpoint {name!r} is not present under {root!r}",
                        hint=f"expected a directory named {name!r} beside rl_agent_config.json",
                    )
                models[name] = (root, name)
        router = laya_module.Router(models=models, device=self.config.device)
        return router

    def _apply_overrides(self, agent: Any) -> None:
        """Set the token budgets and note whether the device was demoted.

        Laya falls back to CPU on an OOM during placement and prints a warning to
        stdout. There is no flag, so the only way to know is to compare the
        device it ended on against the one that was asked for.
        """
        if self.config.max_len is not None:
            agent.cfg["max_len"] = int(self.config.max_len)
        if self.config.head_max_len is not None:
            agent.cfg["head_max_len"] = int(self.config.head_max_len)

        actual = str(getattr(agent, "device", "unknown"))
        if self.config.device and actual != self.config.device:
            if actual.startswith("cpu") and not self.config.device.startswith("cpu"):
                self._degraded = True
                log.warning(
                    "Laya placed the model on %s although %s was requested. Inference will be "
                    "roughly 10-15x slower. On an RTX 50-series card this usually means the "
                    "PyTorch wheel was built without sm_120 support.",
                    actual,
                    self.config.device,
                )

    def _describe(self, name: str, agent: Any) -> Capability:
        cfg = getattr(agent, "cfg", {}) or {}
        # Build a throwaway capability from the agent's live config, which is what
        # matters after an override - the file on disk may now disagree with it.
        from .capability import KNOWN_CHECKPOINTS

        repo, subfolder = KNOWN_CHECKPOINTS.get(name, ("?", None))
        return Capability(
            checkpoint=name,
            repo=repo,
            subfolder=subfolder,
            encoder=cfg.get("encoder"),
            device=str(getattr(agent, "device", "unknown")),
            requested_device=self.config.device,
            degraded=self._degraded,
            max_len=int(cfg.get("max_len", 512)),
            head_max_len=int(cfg.get("head_max_len", 192)),
            head_layers=cfg.get("head_layers"),
            temperature=tuple(float(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])),
            fitted_temperature_buckets=len(cfg.get("temperature_by_options") or {}),
            amp_dtype=cfg.get("amp_dtype"),
        )

    def stop(self) -> None:
        """Release the model and reclaim accelerator memory.

        ``Router.unload`` drops its references but does not empty the CUDA
        allocator cache, so a restart in-process would inherit a fragmented
        allocator. Issue #52 upstream is the same class of problem at 21.7 GB.
        """
        with self._state_lock:
            router = self._router
            self._router = None
            self._loaded = False
        if router is not None:
            try:
                router.unload()
            except Exception:  # noqa: BLE001 - a failed unload must not block shutdown
                log.debug("router.unload raised during shutdown", exc_info=True)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        log.info("model released")

    # -- introspection -------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def capabilities(self) -> Mapping[str, Capability]:
        return dict(self._capabilities)

    def health(self) -> dict[str, Any]:
        """A health report that distinguishes "up" from "useful".

        A server that has loaded a model it cannot run is not healthy, and a
        server that silently demoted itself to CPU is not healthy either - it will
        answer, and it will be wrong about how fast.
        """
        payload: dict[str, Any] = {
            "loaded": self._loaded,
            "degraded": self._degraded,
            "calls": self._calls,
            "failures": self._failures,
            "last_latency_ms": round(self._last_latency_ms, 2),
        }
        if self._started_at is not None:
            payload["uptime_s"] = round(time.time() - self._started_at, 1)
        if self._load_error:
            payload["load_error"] = self._load_error
        payload["checkpoints"] = {
            name: {
                "device": cap.device,
                "max_len": cap.max_len,
                "head_max_len": cap.head_max_len,
                "fitted_temperature_buckets": cap.fitted_temperature_buckets,
            }
            for name, cap in self._capabilities.items()
        }
        if self._calibration.load_error:
            payload["calibration_load_error"] = self._calibration.load_error
        return payload

    def capability_for(self, checkpoint: str) -> Optional[Capability]:
        return self._capabilities.get(checkpoint)

    # -- inference -----------------------------------------------------------

    def ask(self, request: AskRequest) -> AskResponse:
        """Run one batch. The whole public surface of the model host."""
        self.start()

        # Validation and planning happen before the lock: they are pure and a bad
        # request should not queue behind a good one.
        validate_questions(request.questions)
        checkpoint = self._resolve_checkpoint(request)
        capability = self._capabilities.get(checkpoint) or next(
            iter(self._capabilities.values()), None
        )
        if capability is None:  # pragma: no cover - start() guarantees one
            raise UnknownModelError("no checkpoint is loaded")

        plan: BudgetPlan = plan_questions(capability, request.state, request.questions)
        plan.warnings = (*plan.warnings, *self._request_warnings(request, capability))
        if (request.strict or self.config.strict) and not plan.fits:
            raise StateTruncatedError(
                "the state does not fit this checkpoint's token budget",
                hint=plan.recommendation or "shorten the state or raise max_len at startup",
                details=plan.truncation_report() or {},
            )

        if self.config.concurrency <= 1:
            with self._lock:
                return self._run(request, checkpoint, capability, plan)

        # A bounded semaphore keeps a burst from queueing without limit; past the
        # ceiling a caller is told to retry rather than left hanging.
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            raise CapacityError(
                "a forward pass is already running and this host is configured for one at a time",
                hint="raise --concurrency only if you know Laya is not sharing device state here",
            )
        try:
            return self._run(request, checkpoint, capability, plan)
        finally:
            self._lock.release()

    def plan(self, request: AskRequest) -> dict[str, Any]:
        """What :meth:`ask` would do to the state, without a forward pass.

        The same ``plan_questions`` call :meth:`ask` makes, over the same
        capability and the same request, so the two cannot report different
        numbers. Exposed on its own because "will this be cut?" is a question a
        caller should be able to ask for free: a cold host pays a model *load*
        here - loading weights is not inferring with them - and then computes
        nothing at all.
        """
        self.start()
        validate_questions(request.questions)
        checkpoint = self._resolve_checkpoint(request)
        capability = self._capabilities.get(checkpoint) or next(
            iter(self._capabilities.values()), None
        )
        if capability is None:  # pragma: no cover - start() guarantees one
            raise UnknownModelError("no checkpoint is loaded")

        plan: BudgetPlan = plan_questions(capability, request.state, request.questions)
        plan.warnings = (*plan.warnings, *self._request_warnings(request, capability))
        return plan.to_dict()

    def _request_warnings(
        self, request: AskRequest, capability: Capability
    ) -> tuple[str, ...]:
        """Warnings about a request this checkpoint cannot actually serve.

        Both conditions used to pass in silence, which is the one thing this
        package exists not to do.
        """
        warnings: list[str] = []
        if request.lang:
            # Not an oversight to be fixed by forwarding it: Laya's own
            # `system_one(state, questions)` takes no language argument, so there
            # is nowhere for the value to go, and naming `lang` is what selects
            # this path in the first place - the router, which is the only
            # component that reads `lang`, is bypassed by asking for it.
            warnings.append(
                f"lang={request.lang!r} was not applied. Laya's `system_one(state, questions)` "
                "takes no language argument, so the override is dropped; naming `lang` also "
                "takes the explicit path, which bypasses the router - the one component that "
                "does read it. Omit `lang` to let the router route on the text itself."
            )
        caveat = script_caveat(capability, request.state, request.questions)
        if caveat:
            warnings.append(caveat)
        return tuple(warnings)

    def _resolve_checkpoint(self, request: AskRequest) -> str:
        """Which checkpoint will answer.

        An explicit model wins. Otherwise, if only one is loaded, that one. With
        several loaded the router decides, and this reports its choice rather than
        predicting it - the whole point of forwarding ``routing`` is that the
        caller can see the detector's reasoning, including the case where it
        silently treated a Latin-script language as English.
        """
        if request.model:
            try:
                return normalise_checkpoint(request.model)
            except KeyError as exc:
                raise UnknownModelError(
                    f"unknown checkpoint {request.model!r}",
                    hint="known checkpoints are english, multilingual and typed-decisions",
                    cause=exc,
                ) from exc
        if len(self._capabilities) == 1:
            return next(iter(self._capabilities))
        return self.config.model

    def _run(
        self,
        request: AskRequest,
        checkpoint: str,
        capability: Capability,
        plan: BudgetPlan,
    ) -> AskResponse:
        started = time.perf_counter()
        self._calls += 1

        questions = {qid: q.to_laya() for qid, q in request.questions.items()}
        routing: Optional[RoutingInfo] = None

        try:
            if self.config.model_root or request.model or request.lang or request.task:
                # An explicit choice bypasses the router's own reasoning; call the
                # agent directly so the caller gets exactly what it asked for.
                agent = self._router.load(checkpoint)
                kwargs = {}
                if request.task:
                    kwargs["task"] = request.task
                if request.lang:
                    kwargs["lang"] = request.lang
                raw = agent.system_one(request.state, questions)
                routing = RoutingInfo(
                    model=checkpoint, lang=request.lang, reason="explicit model selection"
                )
            else:
                raw = self._router.predict(
                    request.state, questions, **self._route_kwargs(request)
                )
                routing = self._routing_from(raw, checkpoint)
        except LayaMcpError:
            self._failures += 1
            raise
        except Exception as exc:  # noqa: BLE001
            self._failures += 1
            translated = translate(exc)
            if isinstance(translated, OutOfMemoryError):
                # Laya may have already demoted itself; either way the caller
                # should know the host is now in a worse state than it was.
                self._degraded = True
            raise translated from exc

        latency_ms = (time.perf_counter() - started) * 1000
        self._last_latency_ms = latency_ms

        answers = self._normalise_answers(raw, request.questions, checkpoint, capability)
        truncated = plan.truncation_report()

        warnings: list[str] = list(plan.warnings)
        if self._degraded:
            warnings.append(
                "this host is running on CPU because Laya demoted itself after a device error; "
                "expect roughly 10-15x the latency you measured on the accelerator"
            )

        return AskResponse(
            answers=answers,
            model=checkpoint,
            routing=routing,
            usage=Usage(
                input_tokens=int((raw.get("usage") or {}).get("input_tokens", 0)),
                output_tokens=int((raw.get("usage") or {}).get("output_tokens", 0)),
                questions=len(questions),
            ),
            latency_ms=latency_ms,
            truncated=truncated,
            budget=plan.to_dict(),
            device=str(capability.device),
            degraded=self._degraded,
            warnings=tuple(warnings),
        )

    def _route_kwargs(self, request: AskRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if request.task:
            kwargs["task"] = request.task
        if request.lang:
            kwargs["lang"] = request.lang
        return kwargs

    def _routing_from(self, raw: Mapping[str, Any], fallback: str) -> RoutingInfo:
        """Read Laya's routing metadata into our own shape.

        Laya's dict has five keys, not the three the README shows: ``model``,
        ``repo``, ``reason``, ``detection`` and ``workflow``. ``model`` here is
        the checkpoint *name*, whereas the ``model`` field inside the answers
        payload is the constant string ``laya-rl-agent``, which names nothing.
        Both facts are why this is translated rather than forwarded.
        """
        routing = raw.get("routing")
        if not isinstance(routing, Mapping):
            return RoutingInfo(model=fallback)
        return RoutingInfo(
            model=str(routing.get("model", fallback)),
            repo=routing.get("repo"),
            reason=routing.get("reason"),
            detection=routing.get("detection"),
            workflow=routing.get("workflow"),
        )

    def _normalise_answers(
        self,
        raw: Mapping[str, Any],
        questions: Mapping[str, Question],
        checkpoint: str,
        capability: Capability,
    ) -> Mapping[str, Answer]:
        """Turn Laya's per-primitive answer dicts into one shape.

        The three primitives return genuinely different structures - a noul has no
        distribution at all, a score reports string-keyed indices - and flattening
        them into one type is what lets the MCP tools and the sidecar share a
        response schema. A field Laya did not return stays ``None`` rather than
        being filled with a plausible default.
        """
        raw_answers = raw.get("answers") or {}
        entry: Optional[CalibrationEntry] = self._calibration.get(checkpoint)
        out: dict[str, Answer] = {}

        for question_id, question in questions.items():
            payload = raw_answers.get(question_id)
            if not isinstance(payload, Mapping):
                # Laya returned nothing for a question we asked. Report it as an
                # answer with no value rather than inventing one.
                out[question_id] = Answer(question_id=question_id, type=question.type)
                continue

            action = payload.get("action") or {}
            act_probability = action.get("act_probability") if isinstance(action, Mapping) else None
            answer = Answer(
                question_id=question_id,
                type=question.type,
                confidence=_as_float(payload.get("confidence")),
                act_probability=_as_float(act_probability),
            )

            if question.type == "noul":
                probability = _as_float(payload.get("noul"))
                answer.noul = probability
                if probability is not None:
                    answer.band = _band(probability)
            elif question.type == "choice":
                answer.choice = payload.get("choice")
                answer.probabilities = _as_float_map(payload.get("probabilities"))
            else:
                answer.score = _as_float(payload.get("score"))
                legend = payload.get("legend")
                if isinstance(legend, Mapping):
                    answer.legend = {str(k): str(v) for k, v in legend.items()}
                answer.probabilities = _as_float_map(payload.get("probabilities"))

            # Apply a fitted calibration on top of whatever the checkpoint shipped.
            # Only the distribution is rescaled; the reported `confidence` is
            # recomputed from the rescaled distribution so the two cannot disagree.
            if entry is not None and not entry.is_identity() and answer.probabilities:
                temperature = entry.temperature_for(question.type, len(answer.probabilities))
                if abs(temperature - 1.0) > 1e-9:
                    rescaled = _rescale(answer, temperature)
                    answer.probabilities = rescaled

            out[question_id] = answer

        return out


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_float_map(value: Any) -> Optional[dict[str, float]]:
    if not isinstance(value, Mapping):
        return None
    out: dict[str, float] = {}
    for key, raw in value.items():
        number = _as_float(raw)
        if number is not None:
            out[str(key)] = number
    return out or None


def _rescale(answer: Answer, temperature: float) -> dict[str, float]:
    """Apply a temperature to an answer's distribution, keeping its top label.

    A calibration changes probabilities, not the argmax - temperature scaling is
    monotone, so the ranking is preserved by construction. Recomputing the
    selected label anyway would be busywork, so this returns the new distribution
    and stops.
    """
    probabilities = answer.probabilities or {}
    keys = list(probabilities.keys())
    scaled = apply_temperature([probabilities[k] for k in keys], temperature)
    return {k: round(v, 6) for k, v in zip(keys, scaled)}
