"""Tests for geometry utilities.

TODO: Replace with real tests after implementation.
"""

import pytest


class TestEstimateDistance:
    def test_known_car_width(self):
        """Car at 10m should give reasonable pixel width."""
        raise NotImplementedError("Write test after geometry.py is implemented")

    def test_farther_object_is_smaller(self):
        """Same object farther away should have smaller pixel width."""
        raise NotImplementedError("Write test after geometry.py is implemented")


class TestLowPassFilter:
    def test_noisy_measurement_smoothing(self):
        """Filter should reduce jitter in simulated noisy data."""
        raise NotImplementedError("Write test after geometry.py is implemented")
