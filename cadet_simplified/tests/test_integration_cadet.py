"""Integration tests using actual CADET simulations.

These tests require CADET and CADET-Process to be installed.
They will be skipped if the dependencies are not available.
"""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

# Check if CADET-Process is available
try:
    from CADETProcess.processModel import (
        ComponentSystem,
        FlowSheet,
        Inlet,
        Outlet,
        LumpedRateModelWithoutPores,
        Process,
    )
    from CADETProcess.processModel import StericMassAction
    from CADETProcess.simulator import Cadet
    CADETPROCESS_INSTALLED = True
    
    # Try to detect CADET installation
    try:
        simulator = Cadet()
        simulator.check_cadet()
        CADET_AVAILABLE = True
    except Exception:
        CADET_AVAILABLE = False
        
except ImportError:
    CADETPROCESS_INSTALLED = False
    CADET_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not CADET_AVAILABLE,
    reason="CADET or CADET-Process not installed"
)


def create_simple_configs() -> tuple:
    """Create simple ExperimentConfig and ColumnBindingConfig for testing.
    
    Configuration:
    - 2 components: Salt, Protein
    - LumpedRateModelWithoutPores
    - 1 cm column, 1 cm diameter
    - StericMassAction binding
    - Quick gradient elution
    
    Returns
    -------
    tuple
        (experiment_config, column_binding)
    """
    from cadet_simplified.operation_modes import (
        ExperimentConfig,
        ColumnBindingConfig,
        ComponentDefinition,
    )
    
    experiment_config = ExperimentConfig(
        name="simple_lwe_test",
        parameters={
            # Flow rate
            "flow_rate_mL_min": 0.1,  # ~0.1 mL/min
            # Volumes in CV
            "equilibration_cv": 1.0,
            "load_cv": 1.0,
            "wash_cv": 1.0,
            "elution_cv": 3.0,
            "strip_cv": 0.0,
            # Salt concentrations in mM
            "load_salt_mM": 50.0,
            "wash_salt_mM": 50.0,
            "gradient_start_mM": 50.0,
            "gradient_end_mM": 500.0,
            "strip_salt_mM": 500.0,
            # pH
            "ph": 7.0,
            # Per-component load concentrations (protein only, salt is set by load_salt_mM)
            "component_2_load_concentration": 1.0,  # 1 g/L protein
        },
        components=[
            ComponentDefinition(name="Salt", is_salt=True),
            ComponentDefinition(name="Protein", is_salt=False),
        ],
    )
    
    column_binding = ColumnBindingConfig(
        column_model="LumpedRateModelWithoutPores",
        binding_model="StericMassAction",
        column_parameters={
            "length": 0.01,           # 1 cm = 0.01 m
            "diameter": 0.01,         # 1 cm = 0.01 m
            "total_porosity": 0.7,
            "axial_dispersion": 1e-7,  # m²/s
        },
        binding_parameters={
            "is_kinetic": False,
            "capacity": 100.0,         # mM
        },
        component_column_parameters={},
        component_binding_parameters={
            "adsorption_rate": [0.0, 0.1],       # Salt doesn't bind
            "desorption_rate": [0.0, 1.0],
            "characteristic_charge": [0.0, 2.0],
            "steric_factor": [0.0, 1.0],
        },
    )
    
    return experiment_config, column_binding

        
def create_simple_process() -> Any:
    """Create a simple LWE process using cadet_simplified.
    
    Returns
    -------
    Process
        CADET-Process Process object ready for simulation
    """
    from cadet_simplified.operation_modes import get_operation_mode
    
    experiment_config, column_binding = create_simple_configs()
    
    # Get the LWE operation mode
    mode = get_operation_mode("LWE_concentration_based")
    
    # Create the process
    process = mode.create_process(experiment_config, column_binding)
    
    return process


def create_simple_lwe_process() -> Any:
    """Create a simple Load-Wash-Elute process for testing (manual construction).

    Configuration:
    - 2 components: Salt, Protein
    - LumpedRateModelWithoutPores (simplest model)
    - 1 cm column, 1 cm diameter
    - StericMassAction binding
    - Quick gradient elution
    """
    # Component system
    component_system = ComponentSystem(['Salt', 'Protein'])

    # Binding model - StericMassAction
    binding_model = StericMassAction(component_system, name='SMA')
    binding_model.is_kinetic = False
    binding_model.adsorption_rate = [0.0, 0.1]      # Salt doesn't bind
    binding_model.desorption_rate = [0.0, 1]
    binding_model.steric_factor = [0, 1]
    binding_model.characteristic_charge = [0, 2]
    binding_model.capacity = 100         # mM

    # Column - LumpedRateModelWithoutPores
    column = LumpedRateModelWithoutPores(component_system, name='column')
    column.binding_model = binding_model

    # Geometry: 1 cm x 1 cm diameter
    column.length = 0.01          # 1 cm = 0.01 m
    column.diameter = 0.01        # 1 cm = 0.01 m
    column.total_porosity = 0.7
    column.axial_dispersion = 1e-7  # m²/s

    # Initial conditions: column equilibrated with low salt
    column.c = [50.0, 0.0]        # mM - salt at 50 mM, no protein
    column.q = [binding_model.capacity, 0]
    
    # Flow sheet
    inlet = Inlet(component_system, name='inlet')
    outlet = Outlet(component_system, name='outlet')

    flow_sheet = FlowSheet(component_system)
    flow_sheet.add_unit(inlet, feed_inlet=True)
    flow_sheet.add_unit(column)
    flow_sheet.add_unit(outlet, product_outlet=True)

    flow_sheet.add_connection(inlet, column)
    flow_sheet.add_connection(column, outlet)

    # Calculate flow rate
    flow_rate = 2e-9  # m3/s
    inlet.flow_rate = flow_rate

    # Process with simple gradient
    process = Process(flow_sheet, 'simple_lwe_test')

    # Total cycle
    equilibration_duration = 1
    load_duration = 1
    wash_duration = 1
    elution_duration = 3
    cycle_time = (
            equilibration_duration +
            load_duration +
            wash_duration +
            elution_duration
        )

    event_times = {
            'Load': equilibration_duration,
            'Wash': equilibration_duration + load_duration,
            'Elution': equilibration_duration + load_duration + wash_duration,
            'Strip': equilibration_duration + load_duration + wash_duration + elution_duration
        }
    process.cycle_time = cycle_time

    # Define concentrations for each phase
    c_load = [50.0, 1.0]      # Salt 50 mM, Protein 1 mM (= 1 Mol/m3)
    c_wash = [50.0, 0.0]      # Salt 50 mM, no protein
    c_elute_start = [50.0, 0.0]
    c_elute_end = [500.0, 0.0]

    # Gradient slope for elution
    elution_duration = 5
    gradient_slope = [(c_elute_end[0] - c_elute_start[0]) / elution_duration,
                        (c_elute_end[1] - c_elute_start[1]) / elution_duration]

    # Polynomial coefficients for gradient
    c_gradient_poly = [
        [c_elute_start[0], gradient_slope[0], 0.0, 0.0],
        [c_elute_start[1], gradient_slope[1], 0.0, 0.0],
    ]

    # Events
    process.add_event('Load', 'flow_sheet.inlet.c', c_load, time=event_times['Load'])
    process.add_event('Wash', 'flow_sheet.inlet.c', c_wash, time=event_times['Wash'])
    process.add_event('grad_start', 'flow_sheet.inlet.c', c_gradient_poly, event_times['Elution'])
    process.add_event('grad_stop', 'flow_sheet.inlet.c', c_elute_end, time=event_times['Strip'])

    return process


class TestIntegrationSimulation:
    """Integration tests with actual CADET simulations."""
    
    def test_simple_simulation_runs(self):
        """Test that a simple simulation completes successfully."""
        from cadet_simplified.simulation import SimulationRunner
        
        process = create_simple_process()
        runner = SimulationRunner()
        
        result = runner.run(process)
        
        assert result.success, f"Simulation failed: {result.errors}"
        assert result.cadet_result is not None
        assert result.time is not None
        assert len(result.time) > 0
    
    def test_simulation_with_h5_output(self):
        """Test simulation with H5 file preservation."""
        from cadet_simplified.simulation import SimulationRunner
        
        process = create_simple_lwe_process()
        runner = SimulationRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_dir = Path(tmpdir)
            
            result = runner.run(process, h5_dir=h5_dir)
            
            assert result.success
            assert result.h5_path is not None
            assert result.h5_path.exists(), "H5 file was not created"
            assert result.h5_path.stat().st_size > 0, "H5 file is empty"


class TestIntegrationExporter:
    """Integration tests for ResultsExporter with real simulations."""
    
    def test_exporter_full_export(self):
        """Test full export workflow with real simulation."""
        from cadet_simplified.simulation import SimulationRunner
        from cadet_simplified.results import ResultsExporter
        
        experiment_config, column_binding = create_simple_configs()
        process = create_simple_process()
        runner = SimulationRunner()
        result = runner.run(process)
        
        assert result.success, f"Simulation failed: {result.errors}"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ResultsExporter(n_interpolation_points=250)
            excel_path = Path(tmpdir) / "results.xlsx"
            
            output_path = exporter.export_simulation_results(
                results=[result],
                experiment_configs=[experiment_config],
                column_binding=column_binding,
                output_path=excel_path,
            )
            
            # Check Excel file exists
            assert output_path.exists()
            
            # Verify Excel content
            xl = pd.ExcelFile(output_path)
            assert "Parameters" in xl.sheet_names
            assert "simple_lwe_test" in xl.sheet_names
            
            # Check parameters sheet
            params_df = pd.read_excel(output_path, sheet_name="Parameters")
            assert len(params_df) == 1
            assert params_df.iloc[0]["experiment_name"] == "simple_lwe_test"
            assert params_df.iloc[0]["simulation_success"] == True
            
            # Check chromatogram sheet - new column naming
            chrom_df = pd.read_excel(output_path, sheet_name="simple_lwe_test")
            assert len(chrom_df) == 250  # n_interpolation_points
            assert "time" in chrom_df.columns
            assert "Salt" in chrom_df.columns
            assert "Protein" in chrom_df.columns


class TestIntegrationStorage:
    """Integration tests for FileResultsStorage with real simulations."""
    
    def test_storage_round_trip(self):
        """Test save and load workflow with real simulation."""
        from cadet_simplified.simulation import SimulationRunner
        from cadet_simplified.storage import FileResultsStorage
        
        experiment_config, column_binding = create_simple_configs()
        process = create_simple_process()
        runner = SimulationRunner()
        result = runner.run(process)
        
        assert result.success, f"Simulation failed: {result.errors}"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileResultsStorage(tmpdir, n_interpolation_points=100)
            
            # Save
            set_id = storage.save_experiment_set(
                name="integration_test",
                operation_mode="LWE_concentration_based",
                experiments=[experiment_config],
                column_binding=column_binding,
                results=[result],
            )
            
            assert set_id is not None
            
            # List
            df = storage.list_experiments()
            assert len(df) == 1
            assert df.iloc[0]["experiment_name"] == "simple_lwe_test"
            assert df.iloc[0]["has_results"] == True
            assert df.iloc[0]["has_chromatogram"] == True
            
            # Load
            loaded = storage.load_results(set_id)
            assert len(loaded) == 1
            
            loaded_exp = loaded[0]
            assert loaded_exp.experiment_name == "simple_lwe_test"
            assert loaded_exp.result.success is True
            assert loaded_exp.chromatogram_df is not None
            assert len(loaded_exp.chromatogram_df) == 100
            
            # Verify chromatogram has data
            assert "time" in loaded_exp.chromatogram_df.columns
            assert "Salt" in loaded_exp.chromatogram_df.columns
            assert "Protein" in loaded_exp.chromatogram_df.columns
    
    def test_storage_with_h5_files(self):
        """Test that H5 files are stored correctly."""
        from cadet_simplified.simulation import SimulationRunner
        from cadet_simplified.storage import FileResultsStorage
        
        experiment_config, column_binding = create_simple_configs()
        process = create_simple_process()
        runner = SimulationRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            h5_dir = tmpdir / "h5_temp"
            h5_dir.mkdir()
            
            # Run with H5 output
            result = runner.run(process, h5_dir=h5_dir)
            assert result.success
            assert result.h5_path is not None
            
            # Save to storage
            storage = FileResultsStorage(tmpdir / "storage")
            set_id = storage.save_experiment_set(
                name="h5_test",
                operation_mode="LWE_concentration_based",
                experiments=[experiment_config],
                column_binding=column_binding,
                results=[result],
            )
            
            # Check H5 was copied
            h5_stored = tmpdir / "storage" / set_id / "h5" / "simple_lwe_test.h5"
            assert h5_stored.exists()
            assert h5_stored.stat().st_size > 0
    
    def test_batch_simulation_and_storage(self):
        """Test batch simulation followed by storage."""
        from cadet_simplified.simulation import SimulationRunner
        from cadet_simplified.storage import FileResultsStorage
        from cadet_simplified.operation_modes import (
            ExperimentConfig,
            ColumnBindingConfig,
            ComponentDefinition,
            get_operation_mode,
        )
        
        # Create multiple experiment configs with different parameters
        configs = []
        for i in range(3):
            config = ExperimentConfig(
                name=f"batch_exp_{i}",
                parameters={
                    "flow_rate_mL_min": 0.1,
                    "equilibration_cv": 1.0,
                    "load_cv": 1.0,
                    "wash_cv": 1.0,
                    "elution_cv": 3.0 + i,  # Vary elution volume
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
            configs.append(config)
        
        _, column_binding = create_simple_configs()
        
        # Create processes
        mode = get_operation_mode("LWE_concentration_based")
        processes = [mode.create_process(cfg, column_binding) for cfg in configs]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = SimulationRunner()
            
            # Run batch (sequential for simplicity in tests)
            results = runner.run_batch(processes, n_cores=1)
            
            assert len(results) == 3
            assert all(r.success for r in results), f"Some simulations failed: {[r.errors for r in results if not r.success]}"
            
            # Save to storage
            storage = FileResultsStorage(tmpdir, n_interpolation_points=50)
            set_id = storage.save_experiment_set(
                name="batch_test",
                operation_mode="LWE_concentration_based",
                experiments=configs,
                column_binding=column_binding,
                results=results,
            )
            
            # Load and verify
            loaded = storage.load_results(set_id)
            assert len(loaded) == 3
            
            names = {exp.experiment_name for exp in loaded}
            assert names == {"batch_exp_0", "batch_exp_1", "batch_exp_2"}
            
            # All should have chromatograms
            for exp in loaded:
                assert exp.chromatogram_df is not None
                assert len(exp.chromatogram_df) == 50


class TestIntegrationAnalysis:
    """Integration tests for the analysis module with real data."""
    
    def test_simple_analysis_with_real_data(self):
        """Test SimpleChromatogramAnalysis with real simulation data."""
        from cadet_simplified.simulation import SimulationRunner
        from cadet_simplified.storage import FileResultsStorage
        from cadet_simplified.analysis import AnalysisView, get_analysis
        
        experiment_config, column_binding = create_simple_configs()
        process = create_simple_process()
        runner = SimulationRunner()
        result = runner.run(process)
        
        assert result.success
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save to storage
            storage = FileResultsStorage(tmpdir, n_interpolation_points=100)
            set_id = storage.save_experiment_set(
                name="analysis_test",
                operation_mode="LWE_concentration_based",
                experiments=[experiment_config],
                column_binding=column_binding,
                results=[result],
            )
            
            # Load
            loaded = storage.load_results(set_id)
            
            # Run analysis
            view = AnalysisView()
            analysis = get_analysis("simple")
            analysis.run(loaded, view)
            
            # View should have content
            assert not view.is_empty
            assert len(view) > 0


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    
    # Run basic tests
    test = TestIntegrationSimulation()
    test.test_simple_simulation_runs()
    print("test_simple_simulation_runs: PASSED")
    
    test.test_simulation_with_h5_output()
    print("test_simulation_with_h5_output: PASSED")
    
    test2 = TestIntegrationExporter()
    test2.test_exporter_full_export()
    print("test_exporter_full_export: PASSED")
    
    test3 = TestIntegrationStorage()
    test3.test_storage_round_trip()
    print("test_storage_round_trip: PASSED")
    
    print("\nAll integration tests passed!")