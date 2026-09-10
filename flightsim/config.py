from __future__ import annotations

import dataclasses
import pathlib
import tomllib
from typing import Any

import tomli_w


def spec(section: str, default: Any, label: str = "", unit: str = "", key: str | None = None, **extra) -> Any:
    meta = {"section": section, "key": key, "label": label, "unit": unit, **extra}
    factory = default if callable(default) else None
    if factory is not None:
        return dataclasses.field(default_factory=factory, metadata=meta)
    return dataclasses.field(default=default, metadata=meta)


def sections(cls) -> dict[str, list[dataclasses.Field]]:
    out: dict[str, list[dataclasses.Field]] = {}
    for f in dataclasses.fields(cls):
        out.setdefault(f.metadata["section"], []).append(f)
    return out


def toml_key(f: dataclasses.Field) -> str:
    return f.metadata.get("key") or f.name


def from_dict(cls, data: dict):
    values = {}
    for f in dataclasses.fields(cls):
        section = data.get(f.metadata["section"], {})
        if f.metadata.get("table"):
            values[f.name] = list(data.get(f.metadata["section"], []))
        elif isinstance(section, dict) and toml_key(f) in section:
            values[f.name] = section[toml_key(f)]
    return cls(**values)


def to_dict(obj) -> dict:
    data: dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if f.metadata.get("table"):
            data[f.metadata["section"]] = list(value)
        else:
            data.setdefault(f.metadata["section"], {})[toml_key(f)] = value
    return data


def load_toml(cls, path: pathlib.Path):
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as fh:
        return from_dict(cls, tomllib.load(fh))


def save_toml(obj, path: pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        tomli_w.dump(to_dict(obj), fh)
