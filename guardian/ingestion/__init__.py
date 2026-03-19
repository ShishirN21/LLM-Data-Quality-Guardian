"""Data ingestion — loaders for CSV, JSON, and text files."""

from guardian.ingestion.base import BaseLoader, LoaderRegistry, create_default_registry

__all__ = ["BaseLoader", "LoaderRegistry", "create_default_registry"]
