"""Integration tests for CADET simulations.

These tests require CADET to be installed and test:
- Process creation with LWEConcentrationBased operation mode
- Process validation
- Running simulations
- Full workflow from config to results

Note: These tests are marked with @pytest.mark.integration and 
may be skipped if CADET is not available.
"""

# requred if run in e.g. debug mode
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytest
import numpy as np

# Try to import CADET - skip tests if not available
try:
    from CADETProcess.simulator import Cadet
    from CADETProcess.processModel import Process
    CADET_AVAILABLE = True
except ImportError:
    CADET_AVAILABLE = False


def create_simple_configs():
    """Create simple ExperimentConfig and ColumnBindingConfig for testing.
    
    Configuration:
    - 2 components: Salt, Protein
    - LumpedRateModelWithoutPores (simplest model)
    - 1 cm column, 1 cm diameter
    - StericMassAction binding
    - Quick gradient elution
    
    Returns
    -------
    tuple
        (experiment_config, column_binding)
    """
    from cadet_simplified.core import (
        ExperimentConfig,
        ColumnBindingConfig,
        ComponentDefinition,
    )
    
    experiment_config = ExperimentConfig(
        name="simple_lwe_test",
        parameters={
            # Flow rate
            "flow_rate_mL_min": 0.1,
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
            # Per-component load concentrations (protein only)
            "component_2_load_concentration": 1.0,
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
    
    return experiment_config, column_binding


def create_lrmp_configs():
    """Create configs using LumpedRateModelWithPores.
    
    Returns
    -------
    tuple
        (experiment_config, column_binding)
    """
    from cadet_simplified.core import (
        ExperimentConfig,
        ColumnBindingConfig,
        ComponentDefinition,
    )
    
    experiment_config = ExperimentConfig(
        name="lrmp_test",
        parameters={
            "flow_rate_mL_min": 0.1,
            "equilibration_cv": 1.0,
            "load_cv": 1.0,
            "wash_cv": 1.0,
            "elution_cv": 3.0,
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
    
    column_binding = ColumnBindingConfig(
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
    
    return experiment_config, column_binding


def create_simple_process():
    """Create a simple LWE process using cadet_simplified.
    
    Returns
    -------
    Process
        CADET-Process Process object ready for simulation
    """
    from cadet_simplified.operation_modes import get_operation_mode
    
    experiment_config, column_binding = create_simple_configs()
    mode = get_operation_mode("LWE_concentration_based")
    process = mode.create_process(experiment_config, column_binding)
    
    return process


requires_cadet = pytest.mark.skipif(
    not CADET_AVAILABLE,
    reason="CADET not installed"
)

class TestOperationModes:
    """Tests for operation mode functionality."""
    
    def test_list_operation_modes(self):
        """Test listing available operation modes."""
        from cadet_simplified.operation_modes import list_operation_modes
        
        modes = list_operation_modes()
        
        assert isinstance(modes, list)
        assert "LWE_concentration_based" in modes
    
    def test_get_operation_mode(self):
        """Test getting an operation mode by name."""
        from cadet_simplified.operation_modes import get_operation_mode
        
        mode = get_operation_mode("LWE_concentration_based")
        
        assert mode is not None
        assert mode.name == "LWE_concentration_based"
    
    def test_get_invalid_operation_mode(self):
        """Test that invalid mode raises ValueError."""
        from cadet_simplified.operation_modes import get_operation_mode
        
        with pytest.raises(ValueError) as exc_info:
            get_operation_mode("invalid_mode")
        
        assert "Unknown operation mode" in str(exc_info.value)
    
    def test_lwe_experiment_parameters(self):
        """Test that LWE mode defines expected parameters."""
        from cadet_simplified.operation_modes import get_operation_mode
        
        mode = get_operation_mode("LWE_concentration_based")
        params = mode.get_experiment_parameters()
        
        param_names = [p.name for p in params]
        
        assert "flow_rate_mL_min" in param_names
        assert "load_cv" in param_names
        assert "wash_cv" in param_names
        assert "elution_cv" in param_names
        assert "gradient_start_mM" in param_names
        assert "gradient_end_mM" in param_names


@requires_cadet
class TestProcessCreation:
    """Tests for creating CADET processes."""
    
    def test_create_process_returns_process(self):
        """Test that create_process returns a Process object."""
        process = create_simple_process()
        
        assert isinstance(process, Process)
        assert process.name == "simple_lwe_test"
    
    def test_process_has_flow_sheet(self):
        """Test that created process has a flow sheet."""
        process = create_simple_process()
        
        assert process.flow_sheet is not None
        assert len(process.flow_sheet.units) > 0
    
    def test_process_has_component_system(self):
        """Test that created process has correct components."""
        process = create_simple_process()
        
        component_system = process.component_system
        assert component_system is not None
        assert len(component_system.components) == 2
    
    def test_process_has_events(self):
        """Test that created process has events defined."""
        process = create_simple_process()
        
        events = process.events
        assert len(events) > 0
    
    def test_process_cycle_time_positive(self):
        """Test that cycle time is positive."""
        process = create_simple_process()
        
        assert process.cycle_time > 0
    
    def test_create_lrmp_process(self):
        """Test creating process with LumpedRateModelWithPores."""
        from cadet_simplified.operation_modes import get_operation_mode
        
        experiment_config, column_binding = create_lrmp_configs()
        mode = get_operation_mode("LWE_concentration_based")
        process = mode.create_process(experiment_config, column_binding)
        
        assert isinstance(process, Process)
        assert process.name == "lrmp_test"


@requires_cadet
class TestProcessValidation:
    """Tests for process validation."""
    
    def test_validate_valid_process(self):
        """Test that valid process passes validation."""
        from cadet_simplified.simulation import SimulationRunner
        
        process = create_simple_process()
        runner = SimulationRunner()
        
        result = runner.validate(process, "test")
        
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_validation_returns_validation_result(self):
        """Test that validation returns ValidationResult object."""
        from cadet_simplified.simulation import SimulationRunner, ValidationResult
        
        process = create_simple_process()
        runner = SimulationRunner()
        
        result = runner.validate(process, "test")
        
        assert isinstance(result, ValidationResult)
        assert result.experiment_name == "test"


@requires_cadet
class TestSimulationRunning:
    """Tests for running simulations."""
    
    def test_run_simulation_succeeds(self):
        """Test that simulation runs successfully."""
        from cadet_simplified.simulation import SimulationRunner
        
        process = create_simple_process()
        runner = SimulationRunner()
        
        result = runner.run(process)
        
        assert result.success is True
        assert len(result.errors) == 0
    
    def test_run_returns_simulation_result(self):
        """Test that run returns SimulationResultWrapper."""
        from cadet_simplified.simulation import SimulationRunner, SimulationResultWrapper
        
        process = create_simple_process()
        runner = SimulationRunner()
        
        result = runner.run(process)
        
        assert isinstance(result, SimulationResultWrapper)
        assert result.experiment_name == "simple_lwe_test"
    
    def test_run_includes_time_array(self):
        """Test that result includes time array."""
        from cadet_simplified.simulation import SimulationRunner
        
        process = create_simple_process()
        runner = SimulationRunner()
        
        result = runner.run(process)
        
        assert result.time is not None
        assert len(result.time) > 0
    
    def test_run_includes_solution(self):
        """Test that result includes solution dictionary."""
        from cadet_simplified.simulation import SimulationRunner
        
        process = create_simple_process()
        runner = SimulationRunner()
        
        result = runner.run(process)
        
        assert result.solution is not None
        assert "Salt" in result.solution
        assert "Protein" in result.solution
    
    def test_run_records_runtime(self):
        """Test that runtime is recorded."""
        from cadet_simplified.simulation import SimulationRunner
        
        process = create_simple_process()
        runner = SimulationRunner()
        
        result = runner.run(process)
        
        assert result.runtime_seconds > 0
    
    def test_run_includes_cadet_result(self):
        """Test that full CADET result is included."""
        from cadet_simplified.simulation import SimulationRunner
        
        process = create_simple_process()
        runner = SimulationRunner()
        
        result = runner.run(process)
        
        assert result.cadet_result is not None


@requires_cadet
class TestBatchSimulation:
    """Tests for batch simulation running."""
    
    def test_run_batch_multiple_processes(self):
        """Test running multiple processes in batch."""
        from cadet_simplified.simulation import SimulationRunner
        from cadet_simplified.operation_modes import get_operation_mode
        
        exp1, col_bind = create_simple_configs()
        exp2, _ = create_simple_configs()
        exp2.name = "experiment_2"
        
        mode = get_operation_mode("LWE_concentration_based")
        process1 = mode.create_process(exp1, col_bind)
        process2 = mode.create_process(exp2, col_bind)
        
        runner = SimulationRunner()
        results = runner.run_batch([process1, process2], n_cores=1)
        
        assert len(results) == 2
        assert all(r.success for r in results)
    
    def test_run_batch_with_progress_callback(self):
        """Test batch with progress callback."""
        from cadet_simplified.simulation import SimulationRunner
        
        process = create_simple_process()
        runner = SimulationRunner()
        
        callback_calls = []
        
        def callback(current, total, result):
            callback_calls.append((current, total, result.success))
        
        results = runner.run_batch([process], progress_callback=callback, n_cores=1)
        
        assert len(callback_calls) == 1
        assert callback_calls[0] == (1, 1, True)


@requires_cadet
class TestChromatogramInterpolation:
    """Tests for chromatogram data extraction."""
    
    def test_interpolate_chromatogram(self):
        """Test interpolating chromatogram from result."""
        from cadet_simplified.simulation import SimulationRunner
        from cadet_simplified.plotting import interpolate_chromatogram
        
        process = create_simple_process()
        runner = SimulationRunner()
        result = runner.run(process)
        
        df = interpolate_chromatogram(result, n_points=1000)
        
        assert "time" in df.columns
        assert "Salt" in df.columns
        assert "Protein" in df.columns
        assert len(df) == 1000
    
    def test_interpolate_time_in_seconds(self):
        """Test that interpolated time is in seconds."""
        from cadet_simplified.simulation import SimulationRunner
        from cadet_simplified.plotting import interpolate_chromatogram
        
        process = create_simple_process()
        runner = SimulationRunner()
        result = runner.run(process)
        
        df = interpolate_chromatogram(result)
        
        # Time should be in seconds (cycle time is typically in hundreds of seconds)
        assert df["time"].max() > 10  # Should be more than 10 seconds
        assert df["time"].min() == 0.0


@requires_cadet
class TestFullWorkflow:
    """Tests for complete workflow from config to storage."""
    
    def test_complete_workflow(self):
        """Test complete workflow: config -> process -> simulate -> store."""
        import tempfile
        import shutil
        from pathlib import Path
        
        from cadet_simplified.operation_modes import get_operation_mode
        from cadet_simplified.simulation import SimulationRunner
        from cadet_simplified.storage import FileStorage
        
        # Setup
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create configs
            experiment_config, column_binding = create_simple_configs()
            
            # Create process
            mode = get_operation_mode("LWE_concentration_based")
            process = mode.create_process(experiment_config, column_binding)
            
            # Run simulation
            runner = SimulationRunner()
            result = runner.run(process)
            
            assert result.success
            
            # Store results
            storage = FileStorage(temp_dir)
            set_id = storage.save_experiment_set(
                name="Integration Test",
                operation_mode="LWE_concentration_based",
                experiments=[experiment_config],
                column_binding=column_binding,
                results=[result],
            )
            
            # Verify storage
            loaded = storage.load_results(set_id)
            
            assert len(loaded) == 1
            assert loaded[0].experiment_name == "simple_lwe_test"
            assert loaded[0].result.success
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_workflow_with_chromatogram_caching(self):
        """Test that chromatogram is cached during storage."""
        import tempfile
        import shutil
        
        from cadet_simplified.operation_modes import get_operation_mode
        from cadet_simplified.simulation import SimulationRunner
        from cadet_simplified.storage import FileStorage
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            experiment_config, column_binding = create_simple_configs()
            mode = get_operation_mode("LWE_concentration_based")
            process = mode.create_process(experiment_config, column_binding)
            
            runner = SimulationRunner()
            result = runner.run(process)
            
            storage = FileStorage(temp_dir)
            set_id = storage.save_experiment_set(
                name="Test",
                operation_mode="LWE_concentration_based",
                experiments=[experiment_config],
                column_binding=column_binding,
                results=[result],
            )
            
            # Load with chromatogram
            loaded = storage.load_results(set_id, include_chromatogram=True)
            
            assert loaded[0].chromatogram_df is not None
            assert "time" in loaded[0].chromatogram_df.columns
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
