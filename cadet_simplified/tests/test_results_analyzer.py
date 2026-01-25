"""Tests for ResultsAnalyzer.

Minimal test set covering:
- Folder creation with timestamp format
- Excel structure (correct sheets, columns)
- Interpolation output shape
- Parameter export completeness
"""

import re
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import pickle

import numpy as np
import pandas as pd
import pytest


# Fixtures

@pytest.fixture
def mock_simulator():
    """Create a mock CADET simulator."""
    simulator = MagicMock()
    simulator.save_to_h5 = MagicMock()
    return simulator


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


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
    
    # Mock time array
    result.time_complete = np.linspace(0, 100, 1000)
    
    # Mock solution with interpolation
    outlet_solution = MagicMock()
    
    def mock_interpolated(time):
        """Return 2D array: (n_times, n_components)."""
        n_times = len(time)
        n_comp = 2
        # Generate some fake chromatogram data
        return np.column_stack([
            np.sin(time / 10) * 100,  # Salt
            np.exp(-((time - 50) ** 2) / 100) * 50,  # Product (Gaussian peak)
        ])
    
    outlet_solution.outlet.solution_interpolated = mock_interpolated
    result.solution = {"outlet": outlet_solution}
    
    return result


@pytest.fixture
def mock_result_wrapper(mock_cadet_result):
    """Create a mock SimulationResultWrapper."""
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
def mock_experiment_config():
    """Create a mock ExperimentConfig."""
    config = MagicMock()
    config.name = "test_experiment"
    config.parameters = {
        "flow_rate_mL_min": 1.0,
        "load_cv": 5.0,
        "wash_cv": 3.0,
        "elution_cv": 20.0,
        "load_salt_mM": 50.0,
        "gradient_end_mM": 500.0,
    }
    return config


@pytest.fixture
def mock_column_binding():
    """Create a mock ColumnBindingConfig."""
    config = MagicMock()
    config.column_model = "LumpedRateModelWithPores"
    config.binding_model = "StericMassAction"
    config.column_parameters = {
        "length": 0.1,
        "diameter": 0.01,
        "bed_porosity": 0.37,
    }
    config.binding_parameters = {
        "capacity": 100.0,
        "is_kinetic": False,
    }
    config.component_column_parameters = {
        "film_diffusion": [1e-5, 1e-6],
    }
    config.component_binding_parameters = {
        "adsorption_rate": [0.0, 0.1],
        "characteristic_charge": [0.0, 5.0],
    }
    return config


# Tests for folder creation

class TestFolderCreation:
    """Tests for output folder creation."""
    
    def test_get_output_path_creates_folder(self, temp_dir, mock_simulator):
        """Test that get_output_path creates the folder."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        output_path = analyzer.get_output_path("my_study")
        
        assert output_path.exists()
        assert output_path.is_dir()
    
    def test_get_output_path_timestamp_format(self, temp_dir, mock_simulator):
        """Test that folder name has correct timestamp format."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        before = datetime.now()
        output_path = analyzer.get_output_path("test_study")
        after = datetime.now()
        
        # Check format: YYYYMMDD-HH-MM-SS_name
        folder_name = output_path.name
        pattern = r"^\d{8}-\d{2}-\d{2}-\d{2}_test_study$"
        assert re.match(pattern, folder_name), f"Folder name '{folder_name}' doesn't match expected pattern"
        
        # Verify timestamp is reasonable (within a few seconds of test execution)
        timestamp_str = folder_name.split("_")[0]
        folder_time = datetime.strptime(timestamp_str, "%Y%m%d-%H-%M-%S")
        
        # Allow 2 second tolerance for timing edge cases
        before_truncated = before.replace(microsecond=0)
        after_truncated = (after.replace(microsecond=0))
        
        assert before_truncated <= folder_time <= after_truncated or \
               abs((folder_time - before_truncated).total_seconds()) <= 2
    
    def test_get_output_path_sanitizes_name(self, temp_dir, mock_simulator):
        """Test that special characters are sanitized."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        output_path = analyzer.get_output_path("test/study:with<bad>chars")
        
        # Should not contain invalid chars
        assert "/" not in output_path.name
        assert ":" not in output_path.name
        assert "<" not in output_path.name
        assert ">" not in output_path.name


# --- Tests for interpolation ---

class TestInterpolation:
    """Tests for chromatogram interpolation."""
    
    def test_interpolate_chromatogram_shape(
        self, temp_dir, mock_simulator, mock_result_wrapper
    ):
        """Test interpolation returns correct shape."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(
            base_dir=temp_dir,
            simulator=mock_simulator,
            n_interpolation_points=250,
        )
        
        chrom = analyzer.interpolate_chromatogram(mock_result_wrapper)
        
        assert len(chrom.time) == 250
        assert len(chrom.concentrations) == 2  # Salt + Product
        for comp_name, conc in chrom.concentrations.items():
            assert len(conc) == 250
    
    def test_interpolate_chromatogram_custom_points(
        self, temp_dir, mock_simulator, mock_result_wrapper
    ):
        """Test interpolation with custom number of points."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        chrom = analyzer.interpolate_chromatogram(mock_result_wrapper, n_points=100)
        
        assert len(chrom.time) == 100
    
    def test_interpolate_chromatogram_component_names(
        self, temp_dir, mock_simulator, mock_result_wrapper
    ):
        """Test that component names are preserved."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        chrom = analyzer.interpolate_chromatogram(mock_result_wrapper)
        
        assert "Salt" in chrom.concentrations
        assert "Product" in chrom.concentrations
    
    def test_interpolate_chromatogram_time_range(
        self, temp_dir, mock_simulator, mock_result_wrapper
    ):
        """Test that time range matches original."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        chrom = analyzer.interpolate_chromatogram(mock_result_wrapper)
        
        original_time = mock_result_wrapper.cadet_result.time_complete
        assert chrom.time[0] == pytest.approx(original_time.min())
        assert chrom.time[-1] == pytest.approx(original_time.max())


# --- Tests for Excel export ---

class TestExcelExport:
    """Tests for Excel export functionality."""
    
    def test_export_creates_excel(
        self, temp_dir, mock_simulator, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that export creates Excel file."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        output_path = analyzer.export(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            name="test_study",
        )
        
        excel_path = output_path / "results.xlsx"
        assert excel_path.exists()
    
    def test_export_excel_has_parameters_sheet(
        self, temp_dir, mock_simulator, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that Excel has Parameters sheet."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        output_path = analyzer.export(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            name="test_study",
        )
        
        excel_path = output_path / "results.xlsx"
        xl = pd.ExcelFile(excel_path)
        
        assert "Parameters" in xl.sheet_names
    
    def test_export_excel_has_chromatogram_sheets(
        self, temp_dir, mock_simulator, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that Excel has chromatogram sheets."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        output_path = analyzer.export(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            name="test_study",
        )
        
        excel_path = output_path / "results.xlsx"
        xl = pd.ExcelFile(excel_path)
        
        # Should have Parameters + one sheet per experiment
        assert len(xl.sheet_names) >= 2
        assert "test_experiment" in xl.sheet_names
    
    def test_export_excel_parameters_completeness(
        self, temp_dir, mock_simulator, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that parameters sheet contains all required columns."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        output_path = analyzer.export(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            name="test_study",
        )
        
        excel_path = output_path / "results.xlsx"
        params_df = pd.read_excel(excel_path, sheet_name="Parameters")
        
        # Check essential columns
        assert "experiment_name" in params_df.columns
        assert "simulation_success" in params_df.columns
        assert "column_model" in params_df.columns
        assert "binding_model" in params_df.columns
        
        # Check experiment parameters are prefixed
        assert "exp_flow_rate_mL_min" in params_df.columns
        assert "exp_load_cv" in params_df.columns
        
        # Check column parameters are prefixed
        assert "col_length" in params_df.columns
        
        # Check binding parameters are prefixed
        assert "bind_capacity" in params_df.columns
        
        # Check per-component parameters
        assert "col_film_diffusion_comp1" in params_df.columns
        assert "bind_characteristic_charge_comp1" in params_df.columns
    
    def test_export_excel_chromatogram_columns(
        self, temp_dir, mock_simulator, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that chromatogram sheet has correct columns."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        output_path = analyzer.export(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            name="test_study",
        )
        
        excel_path = output_path / "results.xlsx"
        chrom_df = pd.read_excel(excel_path, sheet_name="test_experiment")
        
        assert "time_s" in chrom_df.columns
        assert "c_Salt_mM" in chrom_df.columns
        assert "c_Product_mM" in chrom_df.columns


# Tests for H5 export

class TestH5Export:
    """Tests for H5 file export."""
    
    def test_export_calls_save_to_h5(
        self, temp_dir, mock_simulator, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that export calls simulator.save_to_h5."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        analyzer.export(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            name="test_study",
        )
        
        # Since h5_path is None, should call save_to_h5 for config
        mock_simulator.save_to_h5.assert_called_once()
        
        # Check it was called with correct process
        call_args = mock_simulator.save_to_h5.call_args
        assert call_args[0][0] == mock_result_wrapper.cadet_result.process
    
    def test_export_copies_existing_h5(
        self, temp_dir, mock_simulator, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that existing H5 files are copied instead of recreating."""
        from cadet_simplified.results import ResultsAnalyzer
        
        # Create a fake H5 file
        h5_source = temp_dir / "source.h5"
        h5_source.write_text("fake h5 content")
        mock_result_wrapper.h5_path = h5_source
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        output_path = analyzer.export(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            name="test_study",
        )
        
        # Should NOT call save_to_h5 when h5_path exists
        mock_simulator.save_to_h5.assert_not_called()
        
        # Should copy the file
        copied_h5 = output_path / "test_experiment.h5"
        assert copied_h5.exists()
        assert copied_h5.read_text() == "fake h5 content"


# Tests for pickle export

class TestPickleExport:
    """Tests for pickle backup."""
    
    def test_export_no_pickle_by_default(
        self, temp_dir, mock_simulator, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that pickle is not created by default."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(
            base_dir=temp_dir,
            simulator=mock_simulator,
            save_pickle=False,
        )
        
        output_path = analyzer.export(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            name="test_study",
        )
        
        pickle_path = output_path / "results_backup.pkl"
        assert not pickle_path.exists()
    
    def test_export_creates_pickle_when_enabled(
        self, temp_dir, mock_simulator, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that pickle is created when enabled."""
        from cadet_simplified.results import ResultsAnalyzer
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        
        # Create a simpler result wrapper without MagicMock for pickling
        simple_result = SimulationResultWrapper(
            experiment_name="test_experiment",
            success=True,
            time=np.linspace(0, 100, 100),
            solution={"Salt": np.zeros(100), "Product": np.zeros(100)},
            runtime_seconds=1.5,
            cadet_result=None,  # Can't pickle MagicMock, so use None
            h5_path=None,
        )
        
        analyzer = ResultsAnalyzer(
            base_dir=temp_dir,
            simulator=mock_simulator,
            save_pickle=True,
        )
        
        # Note: This will skip chromatogram export for this result since cadet_result is None
        # But it will still create the pickle file
        output_path = analyzer.get_output_path("test_study")
        analyzer._save_pickle([simple_result], output_path)
        
        pickle_path = output_path / "results_backup.pkl"
        assert pickle_path.exists()
        
        # Verify pickle contains results
        with open(pickle_path, 'rb') as f:
            loaded = pickle.load(f)
        assert len(loaded) == 1
        assert loaded[0].experiment_name == "test_experiment"


# Tests for multiple experiments

class TestMultipleExperiments:
    """Tests for handling multiple experiments."""
    
    def test_export_multiple_experiments(
        self, temp_dir, mock_simulator, mock_cadet_result, mock_column_binding
    ):
        """Test export with multiple experiments."""
        from cadet_simplified.results import ResultsAnalyzer
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        
        # Create multiple results
        results = []
        configs = []
        for i in range(3):
            # Create unique mock for each
            mock_process = MagicMock()
            mock_process.name = f"experiment_{i}"
            comp1 = MagicMock()
            comp1.name = "Salt"
            comp2 = MagicMock()
            comp2.name = "Product"
            mock_process.component_system.components = [comp1, comp2]
            outlet = MagicMock()
            outlet.name = "outlet"
            mock_process.flow_sheet.product_outlets = [outlet]
            
            mock_result = MagicMock()
            mock_result.process = mock_process
            mock_result.time_complete = np.linspace(0, 100, 1000)
            
            def mock_interpolated(time):
                return np.column_stack([
                    np.sin(time / 10) * 100,
                    np.exp(-((time - 50) ** 2) / 100) * 50,
                ])
            
            outlet_solution = MagicMock()
            outlet_solution.outlet.solution_interpolated = mock_interpolated
            mock_result.solution = {"outlet": outlet_solution}
            
            wrapper = SimulationResultWrapper(
                experiment_name=f"experiment_{i}",
                success=True,
                time=np.linspace(0, 100, 1000),
                solution={},
                runtime_seconds=1.0 + i * 0.5,
                cadet_result=mock_result,
            )
            results.append(wrapper)
            
            config = MagicMock()
            config.name = f"experiment_{i}"
            config.parameters = {"flow_rate_mL_min": 1.0 + i * 0.1}
            configs.append(config)
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        output_path = analyzer.export(
            results=results,
            experiment_configs=configs,
            column_binding=mock_column_binding,
            name="multi_study",
        )
        
        excel_path = output_path / "results.xlsx"
        xl = pd.ExcelFile(excel_path)
        
        # Should have Parameters + 3 chromatogram sheets
        assert "Parameters" in xl.sheet_names
        assert "experiment_0" in xl.sheet_names
        assert "experiment_1" in xl.sheet_names
        assert "experiment_2" in xl.sheet_names
        
        # Parameters should have 3 rows
        params_df = pd.read_excel(excel_path, sheet_name="Parameters")
        assert len(params_df) == 3


# Tests for error handling

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_export_requires_name_or_path(
        self, temp_dir, mock_simulator, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that export requires either name or output_path."""
        from cadet_simplified.results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        with pytest.raises(ValueError, match="Either 'name' or 'output_path' must be provided"):
            analyzer.export(
                results=[mock_result_wrapper],
                experiment_configs=[mock_experiment_config],
                column_binding=mock_column_binding,
            )
    
    def test_interpolate_without_cadet_result(self, temp_dir, mock_simulator):
        """Test that interpolation fails gracefully without cadet_result."""
        from cadet_simplified.results import ResultsAnalyzer
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        
        result = SimulationResultWrapper(
            experiment_name="failed_experiment",
            success=False,
            errors=["Simulation failed"],
        )
        
        analyzer = ResultsAnalyzer(base_dir=temp_dir, simulator=mock_simulator)
        
        with pytest.raises(ValueError, match="No cadet_result"):
            analyzer.interpolate_chromatogram(result)
