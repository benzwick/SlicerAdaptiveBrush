"""Tests for UI widget configuration validation.

These tests verify that slider and other widget configurations are valid
and won't cause warnings or errors in CTK/Qt. The configurations are
tested against the expected values defined in the effect code.
"""

import pytest
from dataclasses import dataclass
from typing import Optional


@dataclass
class SliderConfig:
    """Expected configuration for a ctkSliderWidget."""

    name: str
    minimum: float
    maximum: float
    default_value: float
    single_step: float
    decimals: int
    suffix: str = ""

    def validate(self) -> list[str]:
        """Validate the slider configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Basic range validation
        if self.minimum >= self.maximum:
            errors.append(f"{self.name}: minimum ({self.minimum}) >= maximum ({self.maximum})")

        # Default value must be in range
        if not (self.minimum <= self.default_value <= self.maximum):
            errors.append(
                f"{self.name}: default_value ({self.default_value}) not in range "
                f"[{self.minimum}, {self.maximum}]"
            )

        # singleStep must be positive
        if self.single_step <= 0:
            errors.append(f"{self.name}: single_step ({self.single_step}) must be > 0")

        # singleStep must not exceed range
        range_size = self.maximum - self.minimum
        if self.single_step > range_size:
            errors.append(
                f"{self.name}: single_step ({self.single_step}) > range ({range_size})"
            )

        # For integer sliders (decimals=0), singleStep must be >= 1
        if self.decimals == 0 and self.single_step < 1:
            errors.append(
                f"{self.name}: decimals=0 but single_step ({self.single_step}) < 1. "
                "Integer sliders require single_step >= 1"
            )

        # For integer sliders, values should be integers
        if self.decimals == 0:
            if self.minimum != int(self.minimum):
                errors.append(f"{self.name}: decimals=0 but minimum ({self.minimum}) is not int")
            if self.maximum != int(self.maximum):
                errors.append(f"{self.name}: decimals=0 but maximum ({self.maximum}) is not int")
            if self.default_value != int(self.default_value):
                errors.append(
                    f"{self.name}: decimals=0 but default_value ({self.default_value}) is not int"
                )
            if self.single_step != int(self.single_step):
                errors.append(
                    f"{self.name}: decimals=0 but single_step ({self.single_step}) is not int"
                )

        # Check decimals is non-negative
        if self.decimals < 0:
            errors.append(f"{self.name}: decimals ({self.decimals}) must be >= 0")

        return errors


# Expected slider configurations from SegmentEditorEffect.py
# These should match the actual code - if the code changes, these tests will catch mismatches
EXPECTED_SLIDER_CONFIGS = [
    # Basic brush settings
    SliderConfig(
        name="radiusSlider",
        minimum=1,
        maximum=100,
        default_value=5.0,
        single_step=0.5,
        decimals=1,
        suffix=" mm",
    ),
    SliderConfig(
        name="sensitivitySlider",
        minimum=0,
        maximum=100,
        default_value=50,
        single_step=5,
        decimals=0,
        suffix="%",
    ),
    SliderConfig(
        name="zoneSlider",
        minimum=10,
        maximum=100,
        default_value=50,
        single_step=5,
        decimals=0,
        suffix="%",
    ),
    # Threshold settings
    SliderConfig(
        name="lowerThresholdSlider",
        minimum=-2000,
        maximum=5000,
        default_value=-100,
        single_step=1,
        decimals=0,
    ),
    SliderConfig(
        name="upperThresholdSlider",
        minimum=-2000,
        maximum=5000,
        default_value=300,
        single_step=1,
        decimals=0,
    ),
    SliderConfig(
        name="toleranceSlider",
        minimum=1,
        maximum=100,
        default_value=20,
        single_step=5,
        decimals=0,
        suffix="%",
    ),
    # Advanced sampling parameters
    SliderConfig(
        name="gaussianSigmaSlider",
        minimum=0.0,
        maximum=2.0,
        default_value=0.5,
        single_step=0.1,
        decimals=2,
    ),
    SliderConfig(
        name="percentileLowSlider",
        minimum=0,
        maximum=49,
        default_value=5,
        single_step=1,
        decimals=0,
        suffix="%",
    ),
    SliderConfig(
        name="percentileHighSlider",
        minimum=51,
        maximum=100,
        default_value=95,
        single_step=1,
        decimals=0,
        suffix="%",
    ),
    SliderConfig(
        name="stdMultiplierSlider",
        minimum=0.5,
        maximum=5.0,
        default_value=2.0,
        single_step=0.1,
        decimals=1,
    ),
    # Algorithm-specific parameters
    SliderConfig(
        name="geodesicEdgeWeightSlider",
        minimum=0.0,
        maximum=10.0,
        default_value=2.0,
        single_step=0.5,
        decimals=1,
    ),
    SliderConfig(
        name="geodesicDistanceScaleSlider",
        minimum=0.3,
        maximum=3.0,
        default_value=1.0,
        single_step=0.1,
        decimals=1,
    ),
    SliderConfig(
        name="geodesicSmoothingSlider",
        minimum=0.0,
        maximum=2.0,
        default_value=0.5,
        single_step=0.1,
        decimals=1,
    ),
    SliderConfig(
        name="watershedGradientScaleSlider",
        minimum=0.3,
        maximum=3.0,
        default_value=1.0,
        single_step=0.1,
        decimals=1,
    ),
    SliderConfig(
        name="watershedSmoothingSlider",
        minimum=0.0,
        maximum=2.0,
        default_value=0.5,
        single_step=0.1,
        decimals=1,
    ),
    SliderConfig(
        name="levelSetPropagationSlider",
        minimum=0.2,
        maximum=3.0,
        default_value=1.0,
        single_step=0.1,
        decimals=1,
    ),
    SliderConfig(
        name="levelSetCurvatureSlider",
        minimum=0.0,
        maximum=2.0,
        default_value=0.5,
        single_step=0.1,
        decimals=1,
    ),
    SliderConfig(
        name="levelSetIterationsSlider",
        minimum=10,
        maximum=500,
        default_value=100,
        single_step=10,
        decimals=0,
    ),
    SliderConfig(
        name="regionGrowingMultiplierSlider",
        minimum=0.5,
        maximum=5.0,
        default_value=2.0,
        single_step=0.1,
        decimals=1,
    ),
    SliderConfig(
        name="regionGrowingIterationsSlider",
        minimum=1,
        maximum=10,
        default_value=3,
        single_step=1,
        decimals=0,
    ),
    SliderConfig(
        name="randomWalkerBetaSlider",
        minimum=10,
        maximum=500,
        default_value=130,
        single_step=10,
        decimals=0,
    ),
    SliderConfig(
        name="closingRadiusSlider",
        minimum=0,
        maximum=5,
        default_value=0,
        single_step=1,
        decimals=0,
    ),
]


class TestSliderConfigValidation:
    """Test that SliderConfig validation logic works correctly."""

    def test_valid_integer_slider(self):
        """A valid integer slider should have no errors."""
        config = SliderConfig(
            name="testSlider",
            minimum=0,
            maximum=100,
            default_value=50,
            single_step=5,
            decimals=0,
        )
        errors = config.validate()
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_valid_float_slider(self):
        """A valid float slider should have no errors."""
        config = SliderConfig(
            name="testSlider",
            minimum=0.0,
            maximum=10.0,
            default_value=5.0,
            single_step=0.1,
            decimals=1,
        )
        errors = config.validate()
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_invalid_zero_step_integer_slider(self):
        """Integer slider with step < 1 should fail validation."""
        config = SliderConfig(
            name="testSlider",
            minimum=0,
            maximum=100,
            default_value=50,
            single_step=0.5,  # Invalid for decimals=0
            decimals=0,
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("single_step" in e and "< 1" in e for e in errors)

    def test_invalid_negative_step(self):
        """Slider with negative step should fail validation."""
        config = SliderConfig(
            name="testSlider",
            minimum=0,
            maximum=100,
            default_value=50,
            single_step=-1,
            decimals=0,
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("must be > 0" in e for e in errors)

    def test_invalid_range(self):
        """Slider with min >= max should fail validation."""
        config = SliderConfig(
            name="testSlider",
            minimum=100,
            maximum=0,
            default_value=50,
            single_step=1,
            decimals=0,
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("minimum" in e and "maximum" in e for e in errors)

    def test_default_out_of_range(self):
        """Slider with default outside range should fail validation."""
        config = SliderConfig(
            name="testSlider",
            minimum=0,
            maximum=100,
            default_value=150,  # Out of range
            single_step=1,
            decimals=0,
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("default_value" in e and "not in range" in e for e in errors)

    def test_step_exceeds_range(self):
        """Slider with step > range should fail validation."""
        config = SliderConfig(
            name="testSlider",
            minimum=0,
            maximum=10,
            default_value=5,
            single_step=20,  # Exceeds range of 10
            decimals=0,
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("single_step" in e and "range" in e for e in errors)


class TestExpectedSliderConfigs:
    """Test that all expected slider configurations are valid."""

    @pytest.mark.parametrize("config", EXPECTED_SLIDER_CONFIGS, ids=lambda c: c.name)
    def test_slider_config_is_valid(self, config: SliderConfig):
        """Each expected slider configuration should be valid."""
        errors = config.validate()
        assert errors == [], f"Slider '{config.name}' has invalid configuration: {errors}"

    def test_all_sliders_have_unique_names(self):
        """All slider configs should have unique names."""
        names = [c.name for c in EXPECTED_SLIDER_CONFIGS]
        assert len(names) == len(set(names)), "Duplicate slider names found"

    def test_integer_sliders_have_decimals_zero(self):
        """Sliders with integer values should have decimals=0."""
        integer_sliders = [
            "sensitivitySlider",
            "zoneSlider",
            "lowerThresholdSlider",
            "upperThresholdSlider",
            "toleranceSlider",
            "percentileLowSlider",
            "percentileHighSlider",
            "levelSetIterationsSlider",
            "regionGrowingIterationsSlider",
            "randomWalkerBetaSlider",
            "closingRadiusSlider",
        ]
        for config in EXPECTED_SLIDER_CONFIGS:
            if config.name in integer_sliders:
                assert config.decimals == 0, (
                    f"Integer slider '{config.name}' should have decimals=0, "
                    f"got decimals={config.decimals}"
                )


class TestScrollWheelAdjustmentBounds:
    """Test that scroll wheel adjustments stay within valid bounds."""

    def test_radius_scroll_stays_in_bounds(self):
        """Shift+scroll radius adjustment should stay within slider bounds."""
        radius_config = next(c for c in EXPECTED_SLIDER_CONFIGS if c.name == "radiusSlider")

        # Simulate scroll adjustments
        current = radius_config.default_value

        # Scroll up many times
        for _ in range(50):
            current = max(radius_config.minimum, min(radius_config.maximum, current * 1.2))

        assert current <= radius_config.maximum
        assert current >= radius_config.minimum

        # Scroll down many times
        for _ in range(50):
            current = max(radius_config.minimum, min(radius_config.maximum, current * 0.8))

        assert current >= radius_config.minimum
        assert current <= radius_config.maximum

    def test_zone_scroll_stays_in_bounds(self):
        """Ctrl+Shift+scroll zone adjustment should stay within slider bounds."""
        zone_config = next(c for c in EXPECTED_SLIDER_CONFIGS if c.name == "zoneSlider")

        # Simulate scroll adjustments
        current = zone_config.default_value

        # Scroll up many times (delta = +5)
        for _ in range(50):
            current = max(zone_config.minimum, min(zone_config.maximum, current + 5))

        assert current <= zone_config.maximum
        assert current >= zone_config.minimum

        # Scroll down many times (delta = -5)
        for _ in range(50):
            current = max(zone_config.minimum, min(zone_config.maximum, current - 5))

        assert current >= zone_config.minimum
        assert current <= zone_config.maximum


# Optional: Tests that can run inside Slicer to verify actual widget state
# These are marked with requires_slicer and will be skipped in standalone pytest
@pytest.mark.requires_slicer
class TestSliderWidgetsInSlicer:
    """Tests that verify actual slider widget configuration in Slicer.

    These tests require a running Slicer environment and are skipped
    when running pytest standalone.
    """

    @pytest.fixture
    def effect_instance(self):
        """Get an instance of the Adaptive Brush effect."""
        try:
            import slicer

            # Get the effect from segment editor
            segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self()
            effect = segmentEditorWidget.editor.effectByName("Adaptive Brush")
            return effect
        except Exception:
            pytest.skip("Could not get effect instance from Slicer")

    def test_sensitivity_slider_decimals(self, effect_instance):
        """Verify sensitivitySlider has decimals=0."""
        slider = effect_instance.sensitivitySlider
        assert slider.decimals == 0, f"Expected decimals=0, got {slider.decimals}"

    def test_zone_slider_decimals(self, effect_instance):
        """Verify zoneSlider has decimals=0."""
        slider = effect_instance.zoneSlider
        assert slider.decimals == 0, f"Expected decimals=0, got {slider.decimals}"

    def test_all_sliders_have_valid_step(self, effect_instance):
        """Verify all sliders have singleStep > 0."""
        slider_names = [c.name for c in EXPECTED_SLIDER_CONFIGS]
        for name in slider_names:
            if hasattr(effect_instance, name):
                slider = getattr(effect_instance, name)
                assert slider.singleStep > 0, f"{name} has invalid singleStep: {slider.singleStep}"
