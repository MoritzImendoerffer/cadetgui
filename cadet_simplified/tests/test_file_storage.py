"""Tests for FileStorage class.

Tests file-based storage for experiment results including:
- Saving experiment sets with results
- Loading results
- Listing experiments
- Deleting experiment sets
"""

# requred if run in e.g. debug mode
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parents[2]))


import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import pytest
import pandas as pd
import numpy as np

from cadet_simplified.storage import FileStorage, LoadedExperiment, ExperimentInfo
from cadet_simplified.core import ExperimentConfig, ColumnBindingConfig, ComponentDefinition
from cadet_simplified.simulation.runner import SimulationResultWrapper



@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def storage(temp_storage_dir):
    """Create a FileStorage instance with temporary directory."""
    return FileStorage(temp_storage_dir)


@pytest.fixture
def sample_experiment_config():
    """Create a sample ExperimentConfig for testing."""
    return ExperimentConfig(
        name="test_experiment_1",
        parameters={
            "flow_rate_mL_min": 1.0,
            "load_cv": 5.0,
            "wash_cv": 3.0,
            "elution_cv": 10.0,
            "gradient_start_mM": 50.0,
            "gradient_end_mM": 500.0,
        },
        components=[
            ComponentDefinition(name="Salt", is_salt=True),
            ComponentDefinition(name="Product", is_salt=False),
        ],
    )


@pytest.fixture
def sample_column_binding():
    """Create a sample ColumnBindingConfig for testing."""
    return ColumnBindingConfig(
        column_model="LumpedRateModelWithPores",
        binding_model="StericMassAction",
        column_parameters={
            "length": 0.1,
            "diameter": 0.01,
            "bed_porosity": 0.37,
            "particle_porosity": 0.75,
            "particle_radius": 4.5e-5,
            "axial_dispersion": 1e-7,
        },
        binding_parameters={
            "is_kinetic": True,
            "capacity": 100.0,
        },
        component_column_parameters={
            "film_diffusion": [1e-4, 1e-5],
        },
        component_binding_parameters={
            "adsorption_rate": [0.0, 0.1],
            "desorption_rate": [0.0, 1.0],
            "characteristic_charge": [0.0, 5.0],
            "steric_factor": [0.0, 10.0],
        },
    )


@pytest.fixture
def mock_simulation_result():
    """Create a mock SimulationResultWrapper for testing (without CADET)."""
    time_array = np.linspace(0, 100, 500)
    solution = {
        "Salt": np.sin(time_array / 10) * 50 + 100,
        "Product": np.exp(-((time_array - 50) ** 2) / 100) * 10,
    }
    
    return SimulationResultWrapper(
        experiment_name="test_experiment_1",
        success=True,
        time=time_array,
        solution=solution,
        errors=[],
        warnings=[],
        runtime_seconds=1.5,
        cadet_result=None,  # No actual CADET result for unit tests
    )


class TestStorageInit:
    """Tests for FileStorage initialization."""
    
    def test_storage_creates_directory(self, temp_storage_dir):
        """Test that storage creates directory if it doesn't exist."""
        new_dir = temp_storage_dir / "new_storage"
        storage = FileStorage(new_dir)
        assert new_dir.exists()
    
    def test_storage_default_interpolation_points(self, storage):
        """Test default interpolation points setting."""
        assert storage.n_interpolation_points == 2000


class TestSaveExperimentSet:
    """Tests for saving experiment sets."""
    
    def test_save_creates_directory_structure(
        self, storage, sample_experiment_config, sample_column_binding, mock_simulation_result
    ):
        """Test that saving creates proper directory structure."""
        set_id = storage.save_experiment_set(
            name="Test Set",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[mock_simulation_result],
        )
        
        set_dir = storage._get_set_dir(set_id)
        
        assert set_dir.exists()
        assert (set_dir / "config.json").exists()
        assert (set_dir / "results").exists()
        assert (set_dir / "chromatograms").exists()
    
    def test_save_returns_unique_id(
        self, storage, sample_experiment_config, sample_column_binding, mock_simulation_result
    ):
        """Test that saving returns a unique ID."""
        set_id1 = storage.save_experiment_set(
            name="Test Set 1",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[mock_simulation_result],
        )
        
        set_id2 = storage.save_experiment_set(
            name="Test Set 2",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[mock_simulation_result],
        )
        
        assert set_id1 != set_id2
        assert len(set_id1) == 12
        assert len(set_id2) == 12
    
    def test_save_stores_config_json(
        self, storage, sample_experiment_config, sample_column_binding, mock_simulation_result
    ):
        """Test that config.json contains expected fields."""
        import json
        
        set_id = storage.save_experiment_set(
            name="Test Set",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[mock_simulation_result],
            description="Test description",
        )
        
        config_path = storage._get_set_dir(set_id) / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["id"] == set_id
        assert config["name"] == "Test Set"
        assert config["operation_mode"] == "LWE_concentration_based"
        assert config["description"] == "Test description"
        assert "column_binding" in config
        assert "experiments" in config
        assert len(config["experiments"]) == 1
    
    def test_save_stores_result_pickle(
        self, storage, sample_experiment_config, sample_column_binding, mock_simulation_result
    ):
        """Test that results are pickled correctly."""
        set_id = storage.save_experiment_set(
            name="Test Set",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[mock_simulation_result],
        )
        
        pkl_path = storage._get_set_dir(set_id) / "results" / "test_experiment_1.pkl"
        assert pkl_path.exists()
    
    def test_save_skips_failed_results(
        self, storage, sample_experiment_config, sample_column_binding
    ):
        """Test that failed results are not saved."""
        failed_result = SimulationResultWrapper(
            experiment_name="test_experiment_1",
            success=False,
            errors=["Simulation failed"],
        )
        
        set_id = storage.save_experiment_set(
            name="Test Set",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[failed_result],
        )
        
        pkl_path = storage._get_set_dir(set_id) / "results" / "test_experiment_1.pkl"
        assert not pkl_path.exists()


class TestLoadResults:
    """Tests for loading experiment results."""
    
    def test_load_returns_loaded_experiments(
        self, storage, sample_experiment_config, sample_column_binding, mock_simulation_result
    ):
        """Test that load returns LoadedExperiment objects."""
        set_id = storage.save_experiment_set(
            name="Test Set",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[mock_simulation_result],
        )
        
        loaded = storage.load_results(set_id)
        
        assert len(loaded) == 1
        assert isinstance(loaded[0], LoadedExperiment)
        assert loaded[0].experiment_name == "test_experiment_1"
        assert loaded[0].experiment_set_id == set_id
    
    def test_load_includes_experiment_config(
        self, storage, sample_experiment_config, sample_column_binding, mock_simulation_result
    ):
        """Test that loaded experiments include ExperimentConfig."""
        set_id = storage.save_experiment_set(
            name="Test Set",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[mock_simulation_result],
        )
        
        loaded = storage.load_results(set_id)
        
        assert loaded[0].experiment_config is not None
        assert loaded[0].experiment_config.name == "test_experiment_1"
        assert loaded[0].experiment_config.parameters["flow_rate_mL_min"] == 1.0
    
    def test_load_includes_column_binding(
        self, storage, sample_experiment_config, sample_column_binding, mock_simulation_result
    ):
        """Test that loaded experiments include ColumnBindingConfig."""
        set_id = storage.save_experiment_set(
            name="Test Set",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[mock_simulation_result],
        )
        
        loaded = storage.load_results(set_id)
        
        assert loaded[0].column_binding is not None
        assert loaded[0].column_binding.column_model == "LumpedRateModelWithPores"
        assert loaded[0].column_binding.binding_model == "StericMassAction"
    
    def test_load_filter_by_experiment_name(
        self, storage, sample_column_binding, mock_simulation_result
    ):
        """Test loading specific experiments by name."""
        exp1 = ExperimentConfig(
            name="experiment_1",
            parameters={"flow_rate_mL_min": 1.0},
            components=[ComponentDefinition(name="Salt", is_salt=True)],
        )
        exp2 = ExperimentConfig(
            name="experiment_2",
            parameters={"flow_rate_mL_min": 2.0},
            components=[ComponentDefinition(name="Salt", is_salt=True)],
        )
        
        result1 = SimulationResultWrapper(
            experiment_name="experiment_1", success=True, runtime_seconds=1.0
        )
        result2 = SimulationResultWrapper(
            experiment_name="experiment_2", success=True, runtime_seconds=1.0
        )
        
        set_id = storage.save_experiment_set(
            name="Test Set",
            operation_mode="LWE_concentration_based",
            experiments=[exp1, exp2],
            column_binding=sample_column_binding,
            results=[result1, result2],
        )
        
        loaded = storage.load_results(set_id, experiment_names=["experiment_1"])
        
        assert len(loaded) == 1
        assert loaded[0].experiment_name == "experiment_1"
    
    def test_load_nonexistent_set_returns_empty(self, storage):
        """Test that loading non-existent set returns empty list."""
        loaded = storage.load_results("nonexistent_id")
        assert loaded == []


class TestListExperiments:
    """Tests for listing experiments."""
    
    def test_list_returns_dataframe(
        self, storage, sample_experiment_config, sample_column_binding, mock_simulation_result
    ):
        """Test that list_experiments returns a DataFrame."""
        storage.save_experiment_set(
            name="Test Set",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[mock_simulation_result],
        )
        
        df = storage.list_experiments()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
    
    def test_list_includes_expected_columns(
        self, storage, sample_experiment_config, sample_column_binding, mock_simulation_result
    ):
        """Test that listing includes expected columns."""
        storage.save_experiment_set(
            name="Test Set",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[mock_simulation_result],
        )
        
        df = storage.list_experiments()
        
        expected_columns = [
            "experiment_set_id",
            "experiment_set_name",
            "experiment_name",
            "created_at",
            "n_components",
            "column_model",
            "binding_model",
            "has_results",
        ]
        
        for col in expected_columns:
            assert col in df.columns
    
    def test_list_empty_storage(self, storage):
        """Test listing on empty storage."""
        df = storage.list_experiments()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_list_respects_limit(
        self, storage, sample_column_binding
    ):
        """Test that limit parameter works."""
        # Create multiple experiments
        for i in range(5):
            exp = ExperimentConfig(
                name=f"exp_{i}",
                parameters={"flow_rate_mL_min": 1.0},
                components=[ComponentDefinition(name="Salt", is_salt=True)],
            )
            result = SimulationResultWrapper(
                experiment_name=f"exp_{i}", success=True, runtime_seconds=1.0
            )
            storage.save_experiment_set(
                name=f"Set {i}",
                operation_mode="LWE_concentration_based",
                experiments=[exp],
                column_binding=sample_column_binding,
                results=[result],
            )
        
        df = storage.list_experiments(limit=3)
        assert len(df) == 3


class TestDeleteExperimentSet:
    """Tests for deleting experiment sets."""
    
    def test_delete_removes_directory(
        self, storage, sample_experiment_config, sample_column_binding, mock_simulation_result
    ):
        """Test that delete removes the experiment set directory."""
        set_id = storage.save_experiment_set(
            name="Test Set",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[mock_simulation_result],
        )
        
        set_dir = storage._get_set_dir(set_id)
        assert set_dir.exists()
        
        result = storage.delete_experiment_set(set_id)
        
        assert result is True
        assert not set_dir.exists()
    
    def test_delete_nonexistent_returns_false(self, storage):
        """Test that deleting non-existent set returns False."""
        result = storage.delete_experiment_set("nonexistent_id")
        assert result is False
    
    def test_delete_removes_from_listing(
        self, storage, sample_experiment_config, sample_column_binding, mock_simulation_result
    ):
        """Test that deleted set is removed from listing."""
        set_id = storage.save_experiment_set(
            name="Test Set",
            operation_mode="LWE_concentration_based",
            experiments=[sample_experiment_config],
            column_binding=sample_column_binding,
            results=[mock_simulation_result],
        )
        
        # Verify it's in the list
        df_before = storage.list_experiments()
        assert len(df_before) == 1
        
        # Delete it
        storage.delete_experiment_set(set_id)
        
        # Verify it's gone
        df_after = storage.list_experiments()
        assert len(df_after) == 0


class TestHelperFunctions:
    """Tests for helper functions in file_storage module."""
    
    def test_sanitize_filename_removes_invalid_chars(self):
        """Test that invalid characters are replaced."""
        from cadet_simplified.storage.file_storage import _sanitize_filename
        
        assert _sanitize_filename("test<>file") == "test__file"
        assert _sanitize_filename("test:file") == "test_file"
        assert _sanitize_filename("test/file") == "test_file"
    
    def test_sanitize_filename_truncates_long_names(self):
        """Test that long names are truncated."""
        from cadet_simplified.storage.file_storage import _sanitize_filename
        
        long_name = "a" * 150
        result = _sanitize_filename(long_name)
        assert len(result) == 100
    
    def test_generate_id_returns_12_char_string(self):
        """Test that generated ID has correct length."""
        from cadet_simplified.storage.file_storage import _generate_id
        
        id1 = _generate_id("test")
        assert len(id1) == 12
        assert isinstance(id1, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
