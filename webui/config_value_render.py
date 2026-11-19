"""Rendering of one configuration leaf into the string the launch form compares and the command line passes.

A single rule with three consumers: the resolver bootstrap that instantiates the
configuration under the target interpreter and prints its defaults, the
run-config reader that turns a finished run's recorded settings back into
overrides, and the tests that pin the two against each other. The strings are
the ones ConfigCli parses: tuples as comma-separated items, booleans as True or
False, and an unset optional field as the empty string. The module imports
nothing from the project, so the bootstrap subprocess can import it with only
the repository root on ``sys.path``.
"""

from __future__ import annotations

from typing import Union, get_args, get_origin


class ConfigValueRenderer:
    """Renders configuration values and classifies their annotations for the launch form."""

    SCALARS = {bool: "bool", int: "int", float: "float", str: "str"}

    @classmethod
    def render(cls, value) -> str:
        """Returns one configuration leaf rendered as its launch-form string.

        Args:
            value : Configuration value, live from a dataclass or decoded from a run's recorded settings.

        Returns:
            The empty string for None, comma-separated items for a tuple or list, and ``str`` of the value otherwise.
        """
        if value is None:
            return ""
        if isinstance(value, (tuple, list)):
            return ", ".join(cls.render(item) for item in value)
        return str(value)

    @classmethod
    def _members(cls, annotation) -> list:
        """Returns the non-None members of an Optional annotation, or the annotation alone."""
        if get_origin(annotation) is Union:
            return [member for member in get_args(annotation) if member is not type(None)]
        return [annotation]

    @classmethod
    def kind(cls, annotation) -> str:
        """Returns the form type of a field annotation: bool, int, float, str, tuple or the name of an unsupported type."""
        members = cls._members(annotation)
        target  = members[0] if len(members) == 1 else None

        if get_origin(target) is tuple:
            return "tuple"
        if target in cls.SCALARS:
            return cls.SCALARS[target]
        return getattr(get_origin(annotation) or annotation, "__name__", str(annotation)).lower()

    @classmethod
    def editable(cls, annotation) -> bool:
        """Returns whether ConfigCli can set a field of this annotation from the command line."""
        return cls.kind(annotation) in ("bool", "int", "float", "str", "tuple")

    @classmethod
    def nullable(cls, annotation) -> bool:
        """Returns whether a field of this annotation accepts None."""
        return get_origin(annotation) is Union and type(None) in get_args(annotation)
