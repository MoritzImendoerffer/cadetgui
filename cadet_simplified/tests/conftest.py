"""Shared pytest fixtures and configuration.

This module provides common fixtures used across test modules:
- Temporary directories for storage tests
- Sample configurations for CADET tests
- Mock objects for unit testing
"""

import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np


# =============================================================================
# Storage Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def simple_experiment_config():
    """Create a simple ExperimentConfig for unit testing."""
    from cadet_simplified.core import ExperimentConfig, ComponentDefinition
    
    return ExperimentConfig(
        name="test_experiment",
        parameters={
            "flow_rate_mL_min": 1.0,
            "load_cv": 5.0,
            "wash_cv": 3.0,
            "elution_cv": 10.0,
            "equilibration_cv": 2.0,
            "strip_cv": 0.0,
            "load_salt_mM": 50.0,
            "wash_salt_mM": 50.0,
            "gradient_start_mM": 50.0,
            "gradient_end_mM": 500.0,
            "strip_salt_mM": 500.0,
            "ph": 7.0,
            "component_2_load_concentration": 1.0,
        },
        components=[
            ComponentDefinition(name="Salt", is_salt=True),
            ComponentDefinition(name="Protein", is_salt=False),
        ],
    )


@pytest.fixture
def simple_column_binding():
    """Create a simple ColumnBindingConfig for unit testing."""
    from cadet_simplified.core import ColumnBindingConfig
    
    return ColumnBindingConfig(
        column_model="LumpedRateModelWithoutPores",
        binding_model="StericMassAction",
        column_parameters={
            "length": 0.01,
            "diameter": 0.01,
            "total_porosity": 0.7,
            "axial_dispersion": 1e-7,
        },
        binding_parameters={
            "is_kinetic": False,
            "capacity": 100.0,
        },
        component_column_parameters={},
        component_binding_parameters={
            "adsorption_rate": [0.0, 0.1],
            "desorption_rate": [0.0, 1.0],
            "characteristic_charge": [0.0, 2.0],
            "steric_factor": [0.0, 1.0],
        },
    )


@pytest.fixture
def lrmp_column_binding():
    """Create a LumpedRateModelWithPores config for testing."""
    from cadet_simplified.core import ColumnBindingConfig
    
    return ColumnBindingConfig(
        column_model="LumpedRateModelWithPores",
        binding_model="StericMassAction",
        column_parameters={
            "length": 0.01,
            "diameter": 0.01,
            "bed_porosity": 0.37,
            "particle_porosity": 0.75,
            "particle_radius": 4.5e-5,
            "axial_dispersion": 1e-7,
        },
        binding_parameters={
            "is_kinetic": False,
            "capacity": 100.0,
        },
        component_column_parameters={
            "film_diffusion": [1e-4, 1e-5],
            "pore_accessibility": [1.0, 1.0],
        },
        component_binding_parameters={
            "adsorption_rate": [0.0, 0.1],
            "desorption_rate": [0.0, 1.0],
            "characteristic_charge": [0.0, 2.0],
            "steric_factor": [0.0, 1.0],
        },
    )


# =============================================================================
# Mock Result Fixtures
# =============================================================================

@pytest.fixture
def mock_successful_result():
    """Create a mock successful SimulationResultWrapper."""
    from cadet_simplified.simulation.runner import SimulationResultWrapper
    
    time = np.linspace(0, 100, 500)
    solution = {
        "Salt": np.sin(time / 10) * 50 + 100,
        "Protein": np.exp(-((time - 50) ** 2) / 100) * 10,
    }
    
    return SimulationResultWrapper(
        experiment_name="test_experiment",
        success=True,
        time=time,
        solution=solution,
        errors=[],
        warnings=[],
        runtime_seconds=1.5,
        cadet_result=None,
    )


@pytest.fixture
def mock_failed_result():
    """Create a mock failed SimulationResultWrapper."""
    from cadet_simplified.simulation.runner import SimulationResultWrapper
    
    return SimulationResultWrapper(
        experiment_name="failed_experiment",
        success=False,
        errors=["Simulation failed: convergence error"],
        runtime_seconds=0.5,
    )


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
