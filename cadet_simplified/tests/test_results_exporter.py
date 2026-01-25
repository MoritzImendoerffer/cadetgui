"""Tests for ResultsExporter.

Minimal test set covering:
- Interpolation output shape and component names
- Excel structure (correct sheets, columns)
- Parameter export completeness
- Multiple experiment handling
"""

import tempfile
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
    from cadet_simplified.operation_modes import ExperimentConfig, ComponentDefinition
    
    return ExperimentConfig(
        name="test_experiment",
        parameters={
            "flow_rate_mL_min": 1.0,
            "load_cv": 5.0,
            "wash_cv": 3.0,
            "elution_cv": 20.0,
            "load_salt_mM": 50.0,
            "gradient_end_mM": 500.0,
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
            "bed_porosity": 0.37,
        },
        binding_parameters={
            "capacity": 100.0,
            "is_kinetic": False,
        },
        component_column_parameters={
            "film_diffusion": [1e-5, 1e-6],
        },
        component_binding_parameters={
            "adsorption_rate": [0.0, 0.1],
            "characteristic_charge": [0.0, 5.0],
        },
    )


# --- Tests for interpolation ---

class TestInterpolation:
    """Tests for chromatogram interpolation."""
    
    def test_interpolate_chromatogram_shape(self, mock_result_wrapper):
        """Test interpolation returns correct shape."""
        from cadet_simplified.results import ResultsExporter
        
        exporter = ResultsExporter(n_interpolation_points=250)
        
        chrom = exporter.interpolate_chromatogram(mock_result_wrapper)
        
        assert len(chrom.time) == 250
        assert len(chrom.concentrations) == 2  # Salt + Product
        for comp_name, conc in chrom.concentrations.items():
            assert len(conc) == 250
    
    def test_interpolate_chromatogram_custom_points(self, mock_result_wrapper):
        """Test interpolation with custom number of points."""
        from cadet_simplified.results import ResultsExporter
        
        exporter = ResultsExporter()
        
        chrom = exporter.interpolate_chromatogram(mock_result_wrapper, n_points=100)
        
        assert len(chrom.time) == 100
    
    def test_interpolate_chromatogram_component_names(self, mock_result_wrapper):
        """Test that component names are preserved."""
        from cadet_simplified.results import ResultsExporter
        
        exporter = ResultsExporter()
        
        chrom = exporter.interpolate_chromatogram(mock_result_wrapper)
        
        assert "Salt" in chrom.concentrations
        assert "Product" in chrom.concentrations
    
    def test_interpolate_chromatogram_time_range(self, mock_result_wrapper):
        """Test that time range matches original."""
        from cadet_simplified.results import ResultsExporter
        
        exporter = ResultsExporter()
        
        chrom = exporter.interpolate_chromatogram(mock_result_wrapper)
        
        original_time = mock_result_wrapper.cadet_result.time_complete
        assert chrom.time[0] == pytest.approx(original_time.min())
        assert chrom.time[-1] == pytest.approx(original_time.max())
    
    def test_interpolate_chromatogram_to_dataframe(self, mock_result_wrapper):
        """Test conversion to DataFrame."""
        from cadet_simplified.results import ResultsExporter
        
        exporter = ResultsExporter(n_interpolation_points=100)
        
        chrom = exporter.interpolate_chromatogram(mock_result_wrapper)
        df = chrom.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "time" in df.columns
        assert "Salt" in df.columns
        assert "Product" in df.columns


# --- Tests for Excel export ---

class TestExcelExport:
    """Tests for Excel export functionality."""
    
    def test_export_creates_excel(
        self, temp_dir, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that export creates Excel file."""
        from cadet_simplified.results import ResultsExporter
        
        exporter = ResultsExporter()
        excel_path = temp_dir / "results.xlsx"
        
        output_path = exporter.export_simulation_results(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            output_path=excel_path,
        )
        
        assert output_path.exists()
        assert output_path == excel_path
    
    def test_export_excel_has_parameters_sheet(
        self, temp_dir, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that Excel has Parameters sheet."""
        from cadet_simplified.results import ResultsExporter
        
        exporter = ResultsExporter()
        excel_path = temp_dir / "results.xlsx"
        
        exporter.export_simulation_results(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            output_path=excel_path,
        )
        
        xl = pd.ExcelFile(excel_path)
        assert "Parameters" in xl.sheet_names
    
    def test_export_excel_has_chromatogram_sheets(
        self, temp_dir, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that Excel has chromatogram sheets."""
        from cadet_simplified.results import ResultsExporter
        
        exporter = ResultsExporter()
        excel_path = temp_dir / "results.xlsx"
        
        exporter.export_simulation_results(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            output_path=excel_path,
        )
        
        xl = pd.ExcelFile(excel_path)
        
        # Should have Parameters + one sheet per experiment
        assert len(xl.sheet_names) >= 2
        assert "test_experiment" in xl.sheet_names
    
    def test_export_excel_parameters_completeness(
        self, temp_dir, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that parameters sheet contains all required columns."""
        from cadet_simplified.results import ResultsExporter
        
        exporter = ResultsExporter()
        excel_path = temp_dir / "results.xlsx"
        
        exporter.export_simulation_results(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            output_path=excel_path,
        )
        
        params_df = pd.read_excel(excel_path, sheet_name="Parameters")
        
        # Check essential columns
        assert "experiment_name" in params_df.columns
        assert "simulation_success" in params_df.columns
        assert "column_model" in params_df.columns
        assert "binding_model" in params_df.columns
        
        # Check experiment parameters are prefixed with exp_
        assert "exp_flow_rate_mL_min" in params_df.columns
        assert "exp_load_cv" in params_df.columns
        
        # Check column parameters are prefixed with col_
        assert "col_length" in params_df.columns
        
        # Check binding parameters are prefixed with bind_
        assert "bind_capacity" in params_df.columns
        
        # Check per-component parameters use _compN suffix
        assert "col_film_diffusion_comp1" in params_df.columns
        assert "bind_characteristic_charge_comp1" in params_df.columns
    
    def test_export_excel_chromatogram_columns(
        self, temp_dir, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that chromatogram sheet has correct columns."""
        from cadet_simplified.results import ResultsExporter
        
        exporter = ResultsExporter()
        excel_path = temp_dir / "results.xlsx"
        
        exporter.export_simulation_results(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            output_path=excel_path,
        )
        
        chrom_df = pd.read_excel(excel_path, sheet_name="test_experiment")
        
        # New column naming: time, <component_name>
        assert "time" in chrom_df.columns
        assert "Salt" in chrom_df.columns
        assert "Product" in chrom_df.columns
    
    def test_export_creates_parent_directories(
        self, temp_dir, mock_result_wrapper,
        mock_experiment_config, mock_column_binding
    ):
        """Test that export creates parent directories if needed."""
        from cadet_simplified.results import ResultsExporter
        
        exporter = ResultsExporter()
        excel_path = temp_dir / "subdir" / "nested" / "results.xlsx"
        
        output_path = exporter.export_simulation_results(
            results=[mock_result_wrapper],
            experiment_configs=[mock_experiment_config],
            column_binding=mock_column_binding,
            output_path=excel_path,
        )
        
        assert output_path.exists()
        assert output_path.parent.exists()


# --- Tests for multiple experiments ---

class TestMultipleExperiments:
    """Tests for handling multiple experiments."""
    
    def test_export_multiple_experiments(self, temp_dir, mock_column_binding):
        """Test export with multiple experiments."""
        from cadet_simplified.results import ResultsExporter
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        from cadet_simplified.operation_modes import ExperimentConfig, ComponentDefinition
        
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
            
            config = ExperimentConfig(
                name=f"experiment_{i}",
                parameters={"flow_rate_mL_min": 1.0 + i * 0.1},
                components=[
                    ComponentDefinition(name="Salt", is_salt=True),
                    ComponentDefinition(name="Product", is_salt=False),
                ],
            )
            configs.append(config)
        
        exporter = ResultsExporter()
        excel_path = temp_dir / "multi_study.xlsx"
        
        exporter.export_simulation_results(
            results=results,
            experiment_configs=configs,
            column_binding=mock_column_binding,
            output_path=excel_path,
        )
        
        xl = pd.ExcelFile(excel_path)
        
        # Should have Parameters + 3 chromatogram sheets
        assert "Parameters" in xl.sheet_names
        assert "experiment_0" in xl.sheet_names
        assert "experiment_1" in xl.sheet_names
        assert "experiment_2" in xl.sheet_names
        
        # Parameters should have 3 rows
        params_df = pd.read_excel(excel_path, sheet_name="Parameters")
        assert len(params_df) == 3
    
    def test_export_skips_failed_experiments(self, temp_dir, mock_column_binding):
        """Test that failed experiments don't get chromatogram sheets."""
        from cadet_simplified.results import ResultsExporter
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        from cadet_simplified.operation_modes import ExperimentConfig, ComponentDefinition
        
        # One successful, one failed
        results = [
            SimulationResultWrapper(
                experiment_name="success_exp",
                success=True,
                time=np.linspace(0, 100, 100),
                solution={},
                runtime_seconds=1.0,
                cadet_result=None,  # No cadet_result means no chromatogram
            ),
            SimulationResultWrapper(
                experiment_name="failed_exp",
                success=False,
                errors=["Simulation failed"],
                runtime_seconds=0.5,
            ),
        ]
        
        configs = [
            ExperimentConfig(
                name="success_exp",
                parameters={"flow_rate_mL_min": 1.0},
                components=[ComponentDefinition(name="Salt", is_salt=True)],
            ),
            ExperimentConfig(
                name="failed_exp",
                parameters={"flow_rate_mL_min": 1.0},
                components=[ComponentDefinition(name="Salt", is_salt=True)],
            ),
        ]
        
        exporter = ResultsExporter()
        excel_path = temp_dir / "partial.xlsx"
        
        exporter.export_simulation_results(
            results=results,
            experiment_configs=configs,
            column_binding=mock_column_binding,
            output_path=excel_path,
        )
        
        xl = pd.ExcelFile(excel_path)
        
        # Parameters sheet should have both experiments
        params_df = pd.read_excel(excel_path, sheet_name="Parameters")
        assert len(params_df) == 2
        
        # Failed experiment should have errors column populated
        failed_row = params_df[params_df["experiment_name"] == "failed_exp"].iloc[0]
        assert "errors" in params_df.columns
        assert pd.notna(failed_row["errors"])


# --- Tests for error handling ---

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_interpolate_without_cadet_result(self):
        """Test that interpolation fails gracefully without cadet_result."""
        from cadet_simplified.results import ResultsExporter
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        
        result = SimulationResultWrapper(
            experiment_name="failed_experiment",
            success=False,
            errors=["Simulation failed"],
        )
        
        exporter = ResultsExporter()
        
        with pytest.raises(ValueError, match="No cadet_result"):
            exporter.interpolate_chromatogram(result)
    
    def test_interpolate_without_product_outlet(self):
        """Test that interpolation fails without product outlet."""
        from cadet_simplified.results import ResultsExporter
        from cadet_simplified.simulation.runner import SimulationResultWrapper
        
        # Create mock with no product outlets
        mock_process = MagicMock()
        mock_process.flow_sheet.product_outlets = []
        
        mock_cadet_result = MagicMock()
        mock_cadet_result.process = mock_process
        
        result = SimulationResultWrapper(
            experiment_name="no_outlet_experiment",
            success=True,
            cadet_result=mock_cadet_result,
        )
        
        exporter = ResultsExporter()
        
        with pytest.raises(ValueError, match="No product outlet"):
            exporter.interpolate_chromatogram(result)


# --- Tests for backward compatibility (ResultsAnalyzer alias) ---

class TestBackwardCompatibility:
    """Tests for ResultsAnalyzer backward compatibility."""
    
    def test_results_analyzer_deprecation_warning(self, temp_dir):
        """Test that ResultsAnalyzer raises deprecation warning."""
        from cadet_simplified.results import ResultsAnalyzer
        
        with pytest.warns(DeprecationWarning, match="ResultsAnalyzer is deprecated"):
            analyzer = ResultsAnalyzer(base_dir=temp_dir)
    
    def test_results_analyzer_inherits_exporter(self, temp_dir):
        """Test that ResultsAnalyzer inherits from ResultsExporter."""
        from cadet_simplified.results import ResultsAnalyzer, ResultsExporter
        
        with pytest.warns(DeprecationWarning):
            analyzer = ResultsAnalyzer(base_dir=temp_dir)
        
        assert isinstance(analyzer, ResultsExporter)