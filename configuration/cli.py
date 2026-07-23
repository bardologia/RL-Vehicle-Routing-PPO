import sys
from dataclasses import fields, is_dataclass
from typing      import Union, get_args, get_origin


class ConfigCli:
    def __init__(self, config, argv=None):
        self.config = config
        self.argv   = list(sys.argv[1:] if argv is None else argv)

    def _pairs(self):
        if len(self.argv) % 2 != 0:
            raise ValueError(f"Overrides must come as '--section.field value' pairs, got: {self.argv}")

        pairs = []
        for flag, raw in zip(self.argv[0::2], self.argv[1::2]):
            if not flag.startswith("--"):
                raise ValueError(f"Expected an override flag starting with '--', got: {flag!r}")
            pairs.append((flag[2:], raw))

        return pairs

    def _resolve(self, path):
        section_name, _, field_name = path.partition(".")
        if not field_name:
            raise ValueError(f"Override path must be 'section.field', got: {path!r}")

        section = getattr(self.config, section_name, None)
        if section is None or not is_dataclass(section):
            raise ValueError(f"Unknown config section: {section_name!r}")

        specs = {spec.name: spec for spec in fields(section) if spec.init}
        if field_name not in specs:
            raise ValueError(f"Unknown config field: {path!r}")

        return section, specs[field_name]

    def _coerce(self, annotation, raw):
        origin = get_origin(annotation)

        if origin is Union:
            members = [arg for arg in get_args(annotation) if arg is not type(None)]
            if raw.lower() in ("none", "null"):
                return None
            return self._coerce(members[0], raw)

        if origin is tuple:
            members = get_args(annotation)
            parts   = [part.strip() for part in raw.split(",")]

            if len(members) == 2 and members[1] is Ellipsis:
                return tuple(self._coerce(members[0], part) for part in parts)
            if len(parts) != len(members):
                raise ValueError(f"Expected {len(members)} comma-separated values, got: {raw!r}")

            return tuple(self._coerce(member, part) for member, part in zip(members, parts))

        if annotation is bool:
            lowered = raw.lower()
            if lowered in ("true", "1"):
                return True
            if lowered in ("false", "0"):
                return False
            raise ValueError(f"Cannot parse boolean override from {raw!r}")

        if annotation is int:
            return int(raw)
        if annotation is float:
            return float(raw)
        if annotation is str:
            return raw

        raise ValueError(f"Field type {annotation!r} cannot be overridden from the command line")

    def apply(self):
        for path, raw in self._pairs():
            section, spec = self._resolve(path)
            setattr(section, spec.name, self._coerce(spec.type, raw))

        return self.config
