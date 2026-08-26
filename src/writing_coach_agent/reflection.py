"""Execution failure classification used by the re-planning loop."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionDecision:
    action: str
    reason: str


class ExecutionReflector:
    def review_failure(self, tool: str, error: Exception, attempt: int, max_retries: int) -> ExecutionDecision:
        if isinstance(error, (TimeoutError, ConnectionError, RuntimeError)) and attempt <= max_retries:
            return ExecutionDecision("retry", "临时性工具错误且仍有重试预算")
        return ExecutionDecision("replan", f"工具 {tool} 不可用，需要基于已有 Memory 重新规划")
