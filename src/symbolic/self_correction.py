from __future__ import annotations
import dataclasses
from collections.abc import Callable
from .z3_engine import SymbolicSpec, Z3CheckResult, Z3LogicalEngine

ProposeFn = Callable[[str, list["SelfCorrectionAttempt"]], SymbolicSpec]


@dataclasses.dataclass
class SelfCorrectionAttempt:
    attempt_idx: int
    spec: SymbolicSpec
    result: Z3CheckResult


@dataclasses.dataclass
class SelfCorrectionOutcome:
    success: bool
    attempts: list[SelfCorrectionAttempt]
    final_result: Z3CheckResult | None

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)


class SelfCorrectionEngine:

    def __init__(self, max_attempts: int = 5):
        self.max_attempts = max_attempts

    def run(
        self, propose_fn: ProposeFn, initial_context: str = ""
    ) -> SelfCorrectionOutcome:
        history: list[SelfCorrectionAttempt] = []
        context = initial_context
        for attempt_idx in range(1, self.max_attempts + 1):
            spec = propose_fn(context, history)
            result = Z3LogicalEngine.check_spec(spec)
            history.append(
                SelfCorrectionAttempt(attempt_idx=attempt_idx, spec=spec, result=result)
            )
            if result.status == "sat":
                return SelfCorrectionOutcome(
                    success=True, attempts=history, final_result=result
                )
            context = self._build_feedback_context(initial_context, history)
        return SelfCorrectionOutcome(
            success=False,
            attempts=history,
            final_result=history[-1].result if history else None,
        )

    @staticmethod
    def _build_feedback_context(
        initial_context: str, history: list[SelfCorrectionAttempt]
    ) -> str:
        last = history[-1]
        lines = [
            initial_context,
            "",
            f"== Proba {last.attempt_idx} odrzucona przez Z3 ==",
        ]
        lines.append(f"Zaproponowane ograniczenia: {last.spec.constraints}")
        lines.append(f"Werdykt Z3: {last.result.message}")
        lines.append(
            "Popraw specyfikacje tak, aby ograniczenia byly wzajemnie spelnialne (SAT), i sprobuj ponownie."
        )
        return "\n".join(lines)
