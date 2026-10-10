"""Actionable errors for numerical information absent from a model/input."""


class CapabilityError(ValueError):
    """A calculation requires an operator or physical input that is unavailable."""
