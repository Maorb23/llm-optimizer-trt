"""Custom exceptions for llm-optimizer."""


class LLMOptimizerError(Exception):
    """Base exception for all llm-optimizer errors."""


class ServerNotReadyError(LLMOptimizerError):
    pass


class ServerCleanupError(LLMOptimizerError):
    """Owned server processes remain alive; further trials must not run."""
