"""Check that the installed package is discoverable."""

from importlib.metadata import version

import jigsaw_ai


def test_installed_package() -> None:
    assert jigsaw_ai.__name__ == "jigsaw_ai"
    assert version("jigsaw-ai") == "0.1.0"
