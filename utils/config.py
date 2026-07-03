"""Tiny YAML config loader with attribute access and CLI dot-overrides.

    cfg = load_config("config/rnai.yaml")
    cfg = apply_overrides(cfg, ["train.lr=1e-3", "model.depth=4"])
    cfg.train.lr      # attribute access
"""
import yaml


class Cfg(dict):
    """dict with attribute access; nested dicts auto-wrapped."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v

    def get(self, k, default=None):
        v = super().get(k, default)
        return Cfg(v) if isinstance(v, dict) else v


def load_config(path):
    with open(path) as f:
        return Cfg(yaml.safe_load(f))


def _coerce(s):
    """Parse a CLI string into bool/int/float/str."""
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def apply_overrides(cfg, overrides):
    """Apply ['a.b=1', ...] dot-path overrides in place."""
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, Cfg())
        d[parts[-1]] = _coerce(val)
    return cfg
