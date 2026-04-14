"""Tests for SupervisorConfig in substation.config."""

import pytest

import substation.config


class TestSupervisorConfig:

	def test_default_disabled (self, minimal_config_dict):
		"""Supervisor is disabled by default."""
		config = substation.config.validate_config(minimal_config_dict)
		assert config.supervisor.enabled is False
		assert config.supervisor.port == 9004

	def test_enable_with_port (self, minimal_config_dict):
		"""Supervisor can be enabled with a custom port."""
		minimal_config_dict["supervisor"] = {"enabled": True, "port": 8888}
		config = substation.config.validate_config(minimal_config_dict)
		assert config.supervisor.enabled is True
		assert config.supervisor.port == 8888

	def test_invalid_port_rejected (self, minimal_config_dict):
		"""Port outside valid range is rejected."""
		import pydantic
		minimal_config_dict["supervisor"] = {"port": 0}
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)
