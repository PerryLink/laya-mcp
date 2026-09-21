"""The wire contract shared by the sidecar, the MCP server, and every adapter.

One request shape and one response shape, defined once. The MCP tools, the HTTP
sidecar, the DSH plugin and the harness installers all speak this, so a change to
a response field is a change in exactly one file.

The shapes are deliberately close to Laya's own so the mapping is obvious, with
three additions that exist because Laya's raw output is not enough to act on
safely:

``truncated``
    Laya cuts one of two things without saying so. The *state* is cut from the
    end when it overflows; the *options* are cut to a per-option token budget
    when they do not fit. A caller that cannot tell the difference between "the
    model read all of this" and "the model read the first 300 tokens of this"
    will draw the wrong conclusion from a confident answer. So both facts are
    reported.

``budget``
    The plan that was actually used, so a caller can see how much room is left
    and adjust rather than guess.

``confidence_semantics``
    A string naming what ``confidence`` means, because it does not mean what most
    callers assume. See :mod:`laya_mcp.calibration`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence, Union

#: The three decision primitives Laya implements. A Laya question with any other
#: ``type`` raises ``KeyError``, which is why this is validated at the edge.
PRIMITIVES: tuple[str, ...] = ("noul", "choice", "score")

#: Answer type tags, matching Laya's own ``type`` field on each answer.
ANSWER_TYPES: tuple[str, ...] = PRIMITIVES

Primitive = Literal["noul", "choice", "score"]

#: What a state may be. Laya's ``serialize_state`` passes a ``str`` through and
#: ``json.dumps`` everything else, so a dict or list is legal. A list is not
#: documented as a conversation-turn shape anywhere in the source despite the
#: docstring claiming it - it is simply JSON-serialised - so this package treats
#: it as opaque JSON and says so rather than pretending to support a format.
State = Union[str, Mapping[str, Any], Sequence[Any]]


@dataclass
class Question:
    """One typed question.

    ``criteria`` means different things per primitive, and the differences are
    load-bearing:

    * ``choice`` - a mapping of the permitted answer labels to an optional
      description. A plain list of labels is also accepted (Laya converts it),
      but a mapping is preferred because the descriptions are what make the
      labels distinguishable to the model.
    * ``score`` - an **ordered sequence** of level descriptions. A mapping is
      refused rather than silently reordered, because a scale is ordered and a
      mapping is not.
    * ``noul`` - optional, and if present it must be the two-outcome mapping
      ``{"true": ..., "false": ...}``. Laya reads only those two keys. Supplying
      them replaces its generic defaults, and it is worth doing whenever the
      boundary between yes and no is not obvious.
    """

    type: Primitive
    instructions: Union[str, Mapping[str, Any], Sequence[Any]]
    criteria: Optional[Any] = None
    id: Optional[str] = None
    """Assigned by the caller or filled in from the request key."""

    def to_laya(self) -> dict[str, Any]:
        """Render this question the way ``Agent.system_one`` expects it."""
        out: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            out["criteria"] = self.criteria
        return out


@dataclass
class Answer:
    """One answer, normalised across the three primitives.

    Only the fields relevant to the answer's own type are populated, and a field
    that Laya did not return is ``None`` rather than a fabricated default. A
    caller that cannot see an answer should know it is missing; a caller shown an
    invented value will act on it.
    """

    question_id: str
    type: Primitive

    noul: Optional[float] = None
    """Probability that the statement holds, in ``[0, 1]``. ``noul`` only."""

    choice: Optional[str] = None
    """The selected label. ``choice`` only."""

    score: Optional[float] = None
    """Expected level on the rubric; may fall between levels. ``score`` only."""

    legend: Optional[Mapping[str, str]] = None
    """Rubric index (as a string) to level description. ``score`` only."""

    probabilities: Optional[Mapping[str, float]] = None
    """The full distribution. ``choice`` keys are labels; ``score`` keys are
    indices as strings. A ``noul`` has no distribution - it is two outcomes, and
    Laya reports only ``p(true)``."""

    confidence: Optional[float] = None
    """Laya's concentration statistic.

    NOT a probability of being correct. For ``choice`` and ``score`` it is
    ``1 - H(p)/log(k)``, a normalised entropy: it is low whenever probability is
    spread across options, *even when the top option is right*. For ``noul`` it
    is ``max(p, 1-p)``, which is a different quantity entirely. See
    :mod:`laya_mcp.calibration` for what this package does about that.
    """

    act_probability: Optional[float] = None
    """Laya's auxiliary action head output.

    Present on every answer and documented nowhere in the README. Its semantics
    are not specified upstream, so it is surfaced but never used for a decision
    here, and it is named as undocumented rather than given a confident gloss.
    """

    band: Optional[str] = None
    """For ``noul`` only: ``no`` / ``uncertain`` / ``yes``, from the probability.

    Reported alongside ``noul`` rather than instead of it, because a calibrated
    probability is not a decision: 0.51 rendered as a settled ``true`` invites a
    caller to branch on a coin toss.
    """

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"question_id": self.question_id, "type": self.type}
        for key in ("noul", "choice", "score", "legend", "probabilities", "confidence", "band"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = dict(value) if isinstance(value, Mapping) else value
        if self.act_probability is not None:
            payload["act_probability"] = self.act_probability
            payload["act_probability_note"] = (
                "Laya's action head; its semantics are undocumented upstream and it is "
                "not used for any decision here"
            )
        return payload


@dataclass
class Usage:
    """Token accounting.

    Laya reports ``input_tokens`` counted over the *padded* batch, so it
    overstates a small batch. It is forwarded for diagnostics and explicitly not
    presented as billable.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    questions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens_padded": self.input_tokens,
            "output_tokens": self.output_tokens,
            "questions": self.questions,
            "note": "input_tokens counts the padded batch, so it overstates small requests",
        }


@dataclass
class RoutingInfo:
    """Which checkpoint answered, and why.

    ``reason`` is Laya's own explanation when the router chose, which is worth
    forwarding verbatim: it is the only signal that a Latin-script language was
    silently treated as English.
    """

    model: Optional[str] = None
    repo: Optional[str] = None
    reason: Optional[str] = None
    detection: Optional[Mapping[str, Any]] = None
    workflow: Optional[str] = None
    lang: Optional[str] = None
    """Set when the caller pinned the language rather than letting detection decide."""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AskRequest:
    """One batch: a state, the questions asked about it, and optional overrides.

    ``model`` is optional and omitting it is the right default: it lets the
    caller's configured routing decide. Pinning it to a named checkpoint is how
    a caller avoids the English-fallback trap for a language the detector does
    not know.
    """

    state: State
    questions: Mapping[str, Question]
    model: Optional[str] = None
    task: Optional[str] = None
    lang: Optional[str] = None
    strict: bool = False
    """When true, a request whose state would be truncated is refused instead of
    answered. Default false: answer, and report ``truncated``."""


@dataclass
class AskResponse:
    """The result of one batch.

    Carries not just the answers but everything needed to judge whether they
    should be trusted: which checkpoint answered, what was cut, and what the
    confidence number actually means.
    """

    answers: Mapping[str, Answer]
    model: Optional[str] = None
    """The checkpoint that answered. Note that Laya's own payload reports the
    constant string ``laya-rl-agent`` here, which names no checkpoint, so this
    package reports the resolved checkpoint instead."""

    routing: Optional[RoutingInfo] = None
    usage: Optional[Usage] = None
    latency_ms: float = 0.0

    truncated: Optional[dict[str, Any]] = None
    """Present only when something was cut. See :class:`~laya_mcp.planning.BudgetPlan`."""

    budget: Optional[dict[str, Any]] = None
    """The plan actually used, with the remaining headroom."""

    device: Optional[str] = None
    degraded: bool = False
    """True when Laya silently demoted itself to CPU; see
    :attr:`laya_mcp.capability.Capability.degraded`."""

    confidence_semantics: str = (
        "Laya's `confidence` is a concentration statistic over the option distribution "
        "(1 - H(p)/log k for choice and score; max(p, 1-p) for noul). It is NOT the "
        "probability that the answer is correct: it is low whenever probability is spread "
        "out, even when the top option is right. Branch on `noul` directly, and treat "
        "`confidence` as a tie-breaker rather than a probability of correctness."
    )

    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "answers": {k: v.to_dict() for k, v in self.answers.items()},
            "latency_ms": round(self.latency_ms, 3),
            "confidence_semantics": self.confidence_semantics,
        }
        if self.model is not None:
            payload["model"] = self.model
        if self.routing is not None:
            payload["routing"] = self.routing.to_dict()
        if self.usage is not None:
            payload["usage"] = self.usage.to_dict()
        if self.truncated is not None:
            payload["truncated"] = self.truncated
        if self.budget is not None:
            payload["budget"] = self.budget
        if self.device is not None:
            payload["device"] = self.device
        if self.degraded:
            payload["degraded"] = True
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload
