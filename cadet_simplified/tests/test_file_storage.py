"""Tests for FileResultsStorage.

Comprehensive tests covering:
- Saving experiment sets with results
- Loading results by selection
- Chromatogram storage and retrieval
- Listing experiments and experiment sets
- Deletion
- Round-trip data integrity
"""

import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def storage(temp_dir):
    """Create a FileResultsStorage instance."""
    from cadet_simplified.storage import FileResultsStorage
    return FileResultsStorage(temp_dir, n_interpolation_points=100)


@pytest.fixture
def mock_process():
    """Create a mock Process object."""
    process = MagicMock()
    process.name = "test_experiment"
    
    # Mock component system
    comp1 = MagicMock()
    comp1.name = "Salt"
    comp2 = MagicMock()
    comp2.name = "Product"
    process.component_system.components = [comp1, comp2]
    
    # Mock flow sheet with product outlet
    outlet = MagicMock()
    outlet.name = "outlet"
    process.flow_sheet.product_outlets = [outlet]
    
    return process


@pytest.fixture
def mock_cadet_result(mock_process):
    """Create a mock SimulationResults object."""
    result = MagicMock()
    result.process = mock_process
    result.time_complete = np.linspace(0, 100, 1000)
    
    def mock_interpolated(time):
        return np.column_stack([
            np.sin(time / 10) * 100,
            np.exp(-((time - 50) ** 2) / 100) * 50,
        ])
    
    outlet_solution = MagicMock()
    outlet_solution.outlet.solution_interpolated = mock_interpolated
    result.solution = {"outlet": outlet_solution}
    
    return result


@pytest.fixture
def mock_result_wrapper_with_cadet(mock_cadet_result):
    """Create a mock SimulationResultWrapper with cadet_result.
    
    WARNING: This fixture contains MagicMock and CANNOT be pickled.
    Use only for tests that don't involve pickling (e.g., ResultsExporter tests).
    For FileResultsStorage tests, use simple_result_wrapper instead.
    """
    from cadet_simplified.simulation.runner import SimulationResultWrapper
    
    return SimulationResultWrapper(
        experiment_name="test_experiment",
        success=True,
        time=np.linspace(0, 100, 1000),
        solution={"Salt": np.zeros(1000), "Product": np.zeros(1000)},
        runtime_seconds=1.5,
        cadet_result=mock_cadet_result,
        h5_path=None,
    )


@pytest.fixture
def mock_result_wrapper():
    """Create a picklable SimulationResultWrapper (no cadet_result).
    
    This fixture can be pickled and is suitable for FileResultsStorage tests.
    Note: No chromatogram will be generated since cadet_result is None.
    """
    from cadet_simplified.simulation.runner import SimulationResultWrapper
    
    return SimulationResultWrapper(
        experiment_name="test_experiment",
        success=True,
        time=np.linspace(0, 100, 1000),
        solution={"Salt": np.zeros(1000), "Product": np.zeros(1000)},
        runtime_seconds=1.5,
        cadet_result=None,
        h5_path=None,
    )


@pytest.fixture
def mock_experiment_config():
    """Create a mock ExperimentConfig."""
    from cadet_simplified.operation_modes import ExperimentConfig, ComponentDefinition
    
    return ExperimentConfig(
        name="test_experiment",
        parameters={
            "flow_rate_mL_min": 1.0,
            "load_cv": 5.0,
            "wash_cv": 3.0,
            "elution_cv": 20.0,
        },
        components=[
            ComponentDefinition(name="Salt", is_salt=True),
            ComponentDefinition(name="Product", is_salt=False),
        ],
    )


@pytest.fixture
def mock_column_binding():
    """Create a mock ColumnBindingConfig."""
    from cadet_simplified.operation_modes import ColumnBindingConfig
    
    return ColumnBindingConfig(
        column_model="LumpedRateModelWithPores",
        binding_model="StericMassAction",
        column_parameters={
            "length": 0.1,
            "diameter": 0.01,
        },
        binding_parameters={
            "capacity": 100.0,
        },
        component_column_parameters={
            "film_diffusion": [1e-5, 1e-6],
        },
        component_binding_parameters={
            "adsorption_rate": [0.0, 0.1],
        },
    )


@pytest.fixture
def simple_result_wrapper():
    """Create a simple SimulationResultWrapper without mock cadet_result.
    
    This is suitable for pickling tests where we need a real object.
    """
    from cadet_simplified.simulation.runner import SimulationResultWrapper
    
    return SimulationResultWrapper(
        experiment_name="simple_experiment",
        success=True,
        time=np.linspace(0, 100, 100),
        solution={"Salt": np.zeros(100), "Product": np.zeros(100)},
        runtime_seconds=1.0,
        cadet_result=None,
        h5_path=None,
    )


# --- Tests for pending directory ---

class TestPendingDirectory:
    """Tests for the _pending directory functionality."""
    
    def test_get_pending_dir_creates_directory(self, storage, temp_dir):
        """Test that get_pending_dir creates the _pending directory."""
        pending_dir = storage.get_pending_dir()
        
        assert pending_dir.exists()
        assert pending_dir.is_dir()
        assert pending_dir.name == "_pending"
        assert pending_dir.parent == temp_dir
    
    def test_get_pending_dir_is_idempotent(self, storage):
        """Test that calling get_pending_dir multiple times returns same path."""
        pending_dir1 = storage.get_pending_dir()
        pending_dir2 = storage.get_pending_dir()
        
        assert pending_dir1 == pending_dir2
    
    def test_clear_pending_removes_files(self, storage):
        """Test that clear_pending removes all files in _pending."""
        pending_dir = storage.get_pending_dir()
        
        # Create some test files
        (pending_dir / "test1.h5").write_bytes(b"test1")
        (pending_dir / "test2.h5").write_bytes(b"test2")
        (pending_dir / "test3.h5").write_bytes(b"test3")
        
        assert len(list(pending_dir.iterdir())) == 3
        
        # Clear
        count = storage.clear_pending()
        
        assert count == 3
        assert len(list(pending_dir.iterdir())) == 0
    
    def test_clear_pending_empty_dir(self, storage):
        """Test that clear_pending handles empty directory."""
        storage.get_pending_dir()  # Ensure it exists
        count = storage.clear_pending()
        assert count == 0
    
    def test_clear_pending_nonexistent_dir(self, storage):
        """Test that clear_pending handles non-existent directory."""
        # Don't call get_pending_dir(), so it doesn't exist
        count = storage.clear_pending()
        assert count == 0
    
    def test_save_cleans_up_h5_from_pending(
        self, storage, temp_dir, mock_experiment_config, mock_column_binding
    ):
        """Test that save_experiment_set cleans up H5 files from pending."""
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        
        # Create H5 file in pending directory
        pending_dir = storage.get_pending_dir()
        h5_source = pending_dir / "test_experiment.h5"
        h5_source.write_bytes(b"fake h5 content")
        
        result = SimulationResultWrapper(
            experiment_name="test_experiment",
            success=True,
            time=np.linspace(0, 100, 100),
            runtime_seconds=1.0,
            cadet_result=None,
            h5_path=h5_source,
        )
        
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[result],
        )
        
        # H5 should be copied to final location
        h5_dest = temp_dir / set_id / "h5" / "test_experiment.h5"
        assert h5_dest.exists()
        assert h5_dest.read_bytes() == b"fake h5 content"
        
        # Original should be deleted
        assert not h5_source.exists()
    
    def test_list_experiments_excludes_pending(
        self, storage, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that list_experiments doesn't include _pending directory."""
        # Create pending directory with some files
        pending_dir = storage.get_pending_dir()
        (pending_dir / "orphan.h5").write_bytes(b"orphan file")
        
        # Save a real experiment
        storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[mock_result_wrapper],
        )
        
        # List should only show the real experiment, not _pending
        df = storage.list_experiments()
        assert len(df) == 1
        assert "_pending" not in df["experiment_set_name"].values
    
    def test_list_experiment_sets_excludes_pending(
        self, storage, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that list_experiment_sets doesn't include _pending directory."""
        # Create pending directory
        storage.get_pending_dir()
        
        # Save a real experiment
        storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[mock_result_wrapper],
        )
        
        # List should only show the real set
        sets = storage.list_experiment_sets()
        assert len(sets) == 1
        assert sets[0]["name"] == "test_study"


# --- Tests for saving experiment sets ---

class TestSaveExperimentSet:
    """Tests for saving experiment sets."""
    
    def test_save_returns_set_id(
        self, storage, simple_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that save returns a valid set ID."""
        # Use simple result without cadet_result for pickle compatibility
        simple_result_wrapper.experiment_name = mock_experiment_config.name
        
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[simple_result_wrapper],
        )
        
        assert set_id is not None
        assert isinstance(set_id, str)
        assert len(set_id) == 12  # MD5 hash truncated to 12 chars
    
    def test_save_creates_directory_structure(
        self, storage, temp_dir, simple_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that save creates correct directory structure."""
        simple_result_wrapper.experiment_name = mock_experiment_config.name
        
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[simple_result_wrapper],
        )
        
        set_dir = temp_dir / set_id
        assert set_dir.exists()
        assert (set_dir / "config.json").exists()
        assert (set_dir / "chromatograms").is_dir()
        assert (set_dir / "results").is_dir()
        assert (set_dir / "h5").is_dir()
    
    def test_save_creates_config_json(
        self, storage, temp_dir, simple_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that config.json has correct content."""
        import json
        
        simple_result_wrapper.experiment_name = mock_experiment_config.name
        
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[simple_result_wrapper],
            description="Test description",
        )
        
        config_path = temp_dir / set_id / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["id"] == set_id
        assert config["name"] == "test_study"
        assert config["operation_mode"] == "LWE_concentration_based"
        assert config["description"] == "Test description"
        assert len(config["experiments"]) == 1
        assert config["experiments"][0]["name"] == "test_experiment"
        assert config["column_binding"]["column_model"] == "LumpedRateModelWithPores"
    
    def test_save_creates_pickle_for_successful_results(
        self, storage, temp_dir, simple_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that pickle files are created for successful results."""
        simple_result_wrapper.experiment_name = mock_experiment_config.name
        
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[simple_result_wrapper],
        )
        
        pkl_path = temp_dir / set_id / "results" / "test_experiment.pkl"
        assert pkl_path.exists()
        
        # Verify pickle content
        with open(pkl_path, 'rb') as f:
            loaded = pickle.load(f)
        assert loaded.experiment_name == "test_experiment"
        assert loaded.success is True
    
    def test_save_without_cadet_result_skips_chromatogram(
        self, storage, temp_dir, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that chromatogram is skipped when cadet_result is None."""
        # mock_result_wrapper has cadet_result=None, so no chromatogram
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[mock_result_wrapper],
        )
        
        # Pickle should exist
        pkl_path = temp_dir / set_id / "results" / "test_experiment.pkl"
        assert pkl_path.exists()
        
        # Chromatogram should NOT exist (no cadet_result to interpolate)
        chrom_path = temp_dir / set_id / "chromatograms" / "test_experiment.parquet"
        assert not chrom_path.exists()
    
    def test_save_skips_failed_results(
        self, storage, temp_dir, mock_experiment_config, mock_column_binding
    ):
        """Test that failed results don't create pickle/chromatogram files."""
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        
        failed_result = SimulationResultWrapper(
            experiment_name="test_experiment",
            success=False,
            errors=["Simulation failed"],
        )
        
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[failed_result],
        )
        
        # Config should exist
        assert (temp_dir / set_id / "config.json").exists()
        
        # But no pickle or chromatogram
        assert not (temp_dir / set_id / "results" / "test_experiment.pkl").exists()
        assert not (temp_dir / set_id / "chromatograms" / "test_experiment.parquet").exists()
    
    def test_save_copies_h5_file(
        self, storage, temp_dir, mock_experiment_config, mock_column_binding
    ):
        """Test that H5 files are copied if they exist."""
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        
        # Create a fake H5 file
        h5_source = temp_dir / "source.h5"
        h5_source.write_bytes(b"fake h5 content")
        
        result = SimulationResultWrapper(
            experiment_name="test_experiment",
            success=True,
            time=np.linspace(0, 100, 100),
            runtime_seconds=1.0,
            cadet_result=None,
            h5_path=h5_source,
        )
        
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[result],
        )
        
        h5_dest = temp_dir / set_id / "h5" / "test_experiment.h5"
        assert h5_dest.exists()
        assert h5_dest.read_bytes() == b"fake h5 content"


# --- Tests for loading results ---

class TestLoadResults:
    """Tests for loading results."""
    
    def test_load_results_returns_loaded_experiments(
        self, storage, simple_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that load_results returns LoadedExperiment objects."""
        simple_result_wrapper.experiment_name = mock_experiment_config.name
        
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[simple_result_wrapper],
        )
        
        loaded = storage.load_results(set_id)
        
        assert len(loaded) == 1
        assert loaded[0].experiment_name == "test_experiment"
        assert loaded[0].experiment_set_id == set_id
        assert loaded[0].experiment_set_name == "test_study"
    
    def test_load_results_includes_config(
        self, storage, simple_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that loaded experiments include config objects."""
        simple_result_wrapper.experiment_name = mock_experiment_config.name
        
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[simple_result_wrapper],
        )
        
        loaded = storage.load_results(set_id)
        
        assert loaded[0].experiment_config.name == "test_experiment"
        assert loaded[0].experiment_config.parameters["flow_rate_mL_min"] == 1.0
        assert loaded[0].column_binding.column_model == "LumpedRateModelWithPores"
    
    def test_load_results_without_chromatogram(
        self, storage, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that loaded experiments have chromatogram_df=None when not available."""
        # mock_result_wrapper has cadet_result=None, so no chromatogram is saved
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[mock_result_wrapper],
        )
        
        loaded = storage.load_results(set_id)
        
        # Chromatogram should be None since it wasn't generated
        assert loaded[0].chromatogram_df is None
    
    def test_load_results_by_name(
        self, storage, mock_column_binding
    ):
        """Test loading specific experiments by name."""
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        from cadet_simplified.operation_modes import ExperimentConfig, ComponentDefinition
        
        # Create multiple experiments
        experiments = []
        results = []
        for i in range(3):
            exp = ExperimentConfig(
                name=f"exp_{i}",
                parameters={"flow_rate_mL_min": 1.0 + i},
                components=[ComponentDefinition(name="Salt", is_salt=True)],
            )
            experiments.append(exp)
            
            result = SimulationResultWrapper(
                experiment_name=f"exp_{i}",
                success=True,
                time=np.linspace(0, 100, 100),
                runtime_seconds=1.0,
                cadet_result=None,
            )
            results.append(result)
        
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=experiments,
            column_binding=mock_column_binding,
            results=results,
        )
        
        # Load only specific experiments
        loaded = storage.load_results(set_id, experiment_names=["exp_0", "exp_2"])
        
        assert len(loaded) == 2
        names = {exp.experiment_name for exp in loaded}
        assert names == {"exp_0", "exp_2"}
    
    def test_load_results_by_selection(
        self, storage, mock_column_binding
    ):
        """Test load_results_by_selection across multiple sets."""
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        from cadet_simplified.operation_modes import ExperimentConfig, ComponentDefinition
        
        # Create two experiment sets
        set_ids = []
        for s in range(2):
            experiments = []
            results = []
            for i in range(2):
                exp = ExperimentConfig(
                    name=f"set{s}_exp_{i}",
                    parameters={"flow_rate_mL_min": 1.0},
                    components=[ComponentDefinition(name="Salt", is_salt=True)],
                )
                experiments.append(exp)
                
                result = SimulationResultWrapper(
                    experiment_name=f"set{s}_exp_{i}",
                    success=True,
                    time=np.linspace(0, 100, 100),
                    runtime_seconds=1.0,
                    cadet_result=None,
                )
                results.append(result)
            
            set_id = storage.save_experiment_set(
                name=f"study_{s}",
                operation_mode="LWE_concentration_based",
                experiments=experiments,
                column_binding=mock_column_binding,
                results=results,
            )
            set_ids.append(set_id)
        
        # Load selection from both sets
        selections = [
            (set_ids[0], "set0_exp_0"),
            (set_ids[1], "set1_exp_1"),
        ]
        loaded = storage.load_results_by_selection(selections)
        
        assert len(loaded) == 2
        names = {exp.experiment_name for exp in loaded}
        assert names == {"set0_exp_0", "set1_exp_1"}
    
    def test_load_results_empty_set(self, storage):
        """Test loading from non-existent set returns empty list."""
        loaded = storage.load_results("nonexistent_id")
        assert loaded == []


# --- Tests for listing experiments ---

class TestListExperiments:
    """Tests for listing experiments."""
    
    def test_list_experiments_returns_dataframe(
        self, storage, simple_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that list_experiments returns a DataFrame."""
        simple_result_wrapper.experiment_name = mock_experiment_config.name
        
        storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[simple_result_wrapper],
        )
        
        df = storage.list_experiments()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
    
    def test_list_experiments_columns(
        self, storage, simple_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that list_experiments has expected columns."""
        simple_result_wrapper.experiment_name = mock_experiment_config.name
        
        storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[simple_result_wrapper],
        )
        
        df = storage.list_experiments()
        
        expected_cols = [
            "experiment_set_id", "experiment_set_name", "experiment_name",
            "created_at", "n_components", "column_model", "binding_model",
            "has_results", "has_chromatogram",
        ]
        for col in expected_cols:
            assert col in df.columns
    
    def test_list_experiments_shows_file_availability(
        self, storage, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that list shows which files are available."""
        # mock_result_wrapper has cadet_result=None
        storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[mock_result_wrapper],
        )
        
        df = storage.list_experiments()
        
        # Should have pickle (results) but no chromatogram (no cadet_result)
        assert df.iloc[0]["has_results"] == True
        assert df.iloc[0]["has_chromatogram"] == False  # No cadet_result to interpolate
    
    def test_list_experiments_respects_limit(
        self, storage, mock_column_binding
    ):
        """Test that list respects the limit parameter."""
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        from cadet_simplified.operation_modes import ExperimentConfig, ComponentDefinition
        
        # Create multiple experiment sets
        for i in range(5):
            exp = ExperimentConfig(
                name=f"exp_{i}",
                parameters={},
                components=[ComponentDefinition(name="Salt", is_salt=True)],
            )
            result = SimulationResultWrapper(
                experiment_name=f"exp_{i}",
                success=True,
                cadet_result=None,
            )
            storage.save_experiment_set(
                name=f"study_{i}",
                operation_mode="LWE_concentration_based",
                experiments=[exp],
                column_binding=mock_column_binding,
                results=[result],
            )
        
        df = storage.list_experiments(limit=3)
        assert len(df) == 3
    
    def test_list_experiment_sets(
        self, storage, simple_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test listing experiment sets."""
        simple_result_wrapper.experiment_name = mock_experiment_config.name
        
        storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[simple_result_wrapper],
        )
        
        sets = storage.list_experiment_sets()
        
        assert len(sets) == 1
        assert sets[0]["name"] == "test_study"
        assert sets[0]["operation_mode"] == "LWE_concentration_based"
        assert sets[0]["n_experiments"] == 1


# --- Tests for chromatogram retrieval ---

class TestChromatogramRetrieval:
    """Tests for direct chromatogram retrieval."""
    
    def test_get_chromatogram(
        self, storage, temp_dir, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test direct chromatogram retrieval."""
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[mock_result_wrapper],
        )
        
        # Manually create a chromatogram parquet file (since mock has no cadet_result)
        chrom_dir = temp_dir / set_id / "chromatograms"
        chrom_dir.mkdir(parents=True, exist_ok=True)
        
        test_df = pd.DataFrame({
            "time": np.linspace(0, 100, 50),
            "Salt": np.ones(50) * 10.0,
            "Product": np.ones(50) * 5.0,
        })
        test_df.to_parquet(chrom_dir / "test_experiment.parquet", index=False)
        
        # Now test retrieval
        chrom = storage.get_chromatogram(set_id, "test_experiment")
        
        assert chrom is not None
        assert isinstance(chrom, pd.DataFrame)
        assert "time" in chrom.columns
        assert len(chrom) == 50
    
    def test_get_chromatogram_not_found(self, storage):
        """Test chromatogram retrieval for non-existent experiment."""
        chrom = storage.get_chromatogram("nonexistent", "nonexistent")
        assert chrom is None


# --- Tests for deletion ---

class TestDeletion:
    """Tests for experiment set deletion."""
    
    def test_delete_experiment_set(
        self, storage, temp_dir, simple_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test deleting an experiment set."""
        simple_result_wrapper.experiment_name = mock_experiment_config.name
        
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[simple_result_wrapper],
        )
        
        # Verify it exists
        assert (temp_dir / set_id).exists()
        
        # Delete
        result = storage.delete_experiment_set(set_id)
        
        assert result is True
        assert not (temp_dir / set_id).exists()
    
    def test_delete_nonexistent_returns_false(self, storage):
        """Test deleting non-existent set returns False."""
        result = storage.delete_experiment_set("nonexistent_id")
        assert result is False


# --- Tests for parallel loading ---

class TestParallelLoading:
    """Tests for parallel loading functionality."""
    
    def test_load_results_parallel(
        self, storage, mock_column_binding
    ):
        """Test loading results with multiple workers."""
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        from cadet_simplified.operation_modes import ExperimentConfig, ComponentDefinition
        
        # Create multiple experiments
        experiments = []
        results = []
        for i in range(5):
            exp = ExperimentConfig(
                name=f"exp_{i}",
                parameters={"flow_rate_mL_min": 1.0 + i},
                components=[ComponentDefinition(name="Salt", is_salt=True)],
            )
            experiments.append(exp)
            
            result = SimulationResultWrapper(
                experiment_name=f"exp_{i}",
                success=True,
                time=np.linspace(0, 100, 100),
                runtime_seconds=1.0,
                cadet_result=None,
            )
            results.append(result)
        
        set_id = storage.save_experiment_set(
            name="test_study",
            operation_mode="LWE_concentration_based",
            experiments=experiments,
            column_binding=mock_column_binding,
            results=results,
        )
        
        # Load with multiple workers
        loaded = storage.load_results(set_id, n_workers=2)
        
        assert len(loaded) == 5
        names = {exp.experiment_name for exp in loaded}
        assert names == {f"exp_{i}" for i in range(5)}


# --- Tests for data integrity ---

class TestDataIntegrity:
    """Tests for round-trip data integrity."""
    
    def test_experiment_config_round_trip(
        self, storage, simple_result_wrapper, mock_column_binding
    ):
        """Test that ExperimentConfig survives save/load round-trip."""
        from cadet_simplified.operation_modes import ExperimentConfig, ComponentDefinition
        
        original_config = ExperimentConfig(
            name="integrity_test",
            parameters={
                "flow_rate_mL_min": 1.5,
                "load_cv": 5.0,
                "some_float": 3.14159,
                "some_int": 42,
            },
            components=[
                ComponentDefinition(name="Salt", is_salt=True),
                ComponentDefinition(name="Protein", is_salt=False, molecular_weight=50000.0),
            ],
        )
        simple_result_wrapper.experiment_name = original_config.name
        
        set_id = storage.save_experiment_set(
            name="integrity_study",
            operation_mode="LWE_concentration_based",
            experiments=[original_config],
            column_binding=mock_column_binding,
            results=[simple_result_wrapper],
        )
        
        loaded = storage.load_results(set_id)
        loaded_config = loaded[0].experiment_config
        
        assert loaded_config.name == original_config.name
        assert loaded_config.parameters["flow_rate_mL_min"] == 1.5
        assert loaded_config.parameters["some_float"] == pytest.approx(3.14159)
        assert loaded_config.parameters["some_int"] == 42
        assert len(loaded_config.components) == 2
        assert loaded_config.components[0].name == "Salt"
        assert loaded_config.components[0].is_salt is True
    
    def test_column_binding_round_trip(
        self, storage, simple_result_wrapper, mock_experiment_config
    ):
        """Test that ColumnBindingConfig survives save/load round-trip."""
        from cadet_simplified.operation_modes import ColumnBindingConfig
        
        original_binding = ColumnBindingConfig(
            column_model="GeneralRateModel",
            binding_model="StericMassAction",
            column_parameters={
                "length": 0.15,
                "diameter": 0.025,
            },
            binding_parameters={
                "capacity": 150.0,
                "is_kinetic": True,
            },
            component_column_parameters={
                "film_diffusion": [1e-5, 2e-5, 3e-5],
            },
            component_binding_parameters={
                "adsorption_rate": [0.0, 0.1, 0.2],
                "characteristic_charge": [0.0, 3.0, 5.0],
            },
        )
        simple_result_wrapper.experiment_name = mock_experiment_config.name
        
        set_id = storage.save_experiment_set(
            name="binding_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=original_binding,
            results=[simple_result_wrapper],
        )
        
        loaded = storage.load_results(set_id)
        loaded_binding = loaded[0].column_binding
        
        assert loaded_binding.column_model == "GeneralRateModel"
        assert loaded_binding.binding_model == "StericMassAction"
        assert loaded_binding.column_parameters["length"] == pytest.approx(0.15)
        assert loaded_binding.binding_parameters["capacity"] == pytest.approx(150.0)
        assert loaded_binding.component_column_parameters["film_diffusion"] == pytest.approx([1e-5, 2e-5, 3e-5])
    
    def test_simulation_result_round_trip(
        self, storage, mock_experiment_config, mock_column_binding
    ):
        """Test that SimulationResultWrapper survives save/load round-trip."""
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        
        original_result = SimulationResultWrapper(
            experiment_name=mock_experiment_config.name,
            success=True,
            time=np.linspace(0, 100, 50),
            solution={"Salt": np.ones(50) * 10.0, "Product": np.ones(50) * 5.0},
            errors=[],
            warnings=["Some warning"],
            runtime_seconds=2.5,
            cadet_result=None,
            h5_path=None,
        )
        
        set_id = storage.save_experiment_set(
            name="result_study",
            operation_mode="LWE_concentration_based",
            experiments=[mock_experiment_config],
            column_binding=mock_column_binding,
            results=[original_result],
        )
        
        loaded = storage.load_results(set_id)
        loaded_result = loaded[0].result
        
        assert loaded_result.experiment_name == mock_experiment_config.name
        assert loaded_result.success is True
        assert loaded_result.runtime_seconds == pytest.approx(2.5)
        assert loaded_result.warnings == ["Some warning"]
        np.testing.assert_array_almost_equal(loaded_result.time, np.linspace(0, 100, 50))