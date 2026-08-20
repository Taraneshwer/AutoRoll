"""
Unit tests for configuration management.
"""

from app.core.config import Settings, get_settings


def test_settings_defaults():
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.VECTOR_DIMENSION == 512
    assert settings.MODEL_VERSION == "arcface_iresnet50_v1"
    assert settings.SERVER_PORT == 8000
