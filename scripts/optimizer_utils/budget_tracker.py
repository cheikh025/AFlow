import asyncio


class BudgetTracker:
    """
    Tracks combined token usage (search + execution) against a hard budget.

    Usage
    -----
    - Call consume(n) after every LLM call with the number of tokens used.
    - Check is_exceeded() / exceeded.is_set() before starting new work.
    - When limit is None the tracker is a no-op (unlimited).
    """

    def __init__(self, token_budget: int | None):
        self.limit = token_budget
        self._used = 0
        self.exceeded = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consume(self, tokens: int) -> None:
        """Add *tokens* to the running total and set exceeded if over limit.

        Synchronous and safe to call from any asyncio coroutine.
        """
        if self.limit is None or tokens <= 0:
            return
        self._used += tokens
        if self._used >= self.limit:
            self.exceeded.set()

    def is_exceeded(self) -> bool:
        return self.limit is not None and self.exceeded.is_set()

    @property
    def used(self) -> int:
        return self._used
