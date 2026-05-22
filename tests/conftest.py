"""Pytest config — adds the project root to sys.path so tests can do
``from notebooks import helpers`` and ``import template_repo`` regardless of
where pytest is invoked from.
"""
import sys
from pathlib import Path

import pytest
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def local_registry(tmp_path: Path) -> ObjectStoreRegistry:
    """Resolves file:// URLs under ``tmp_path`` to a LocalStore."""
    return ObjectStoreRegistry({f"file://{tmp_path}": LocalStore()})
