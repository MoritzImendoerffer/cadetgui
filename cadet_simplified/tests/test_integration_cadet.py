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
    """Create a simple Load-Wash-Elute process for testing.

    Configuration:
    - 2 components: Salt, Protein
    - LumpedRateModelWithoutPores (simplest model)
    - 1 cm column, 1 cm diameter
    - Langmuir binding (simple)
    - Quick gradient elution
    """
    # Component system
    component_system = ComponentSystem(['Salt', 'Protein'])

    # Binding model - Langmuir
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

    # Total cycle: 600 seconds (10 min)
    # - Load: 0-120s (2 min) - protein at 1 g/L, salt at 50 mM
    # - Wash: 120-240s (2 min) - salt at 50 mM
    # - Elute: 240-480s (4 min) - gradient 50 -> 500 mM
    # - Strip: 480-600s (2 min) - salt at 500 mM
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
    #c_strip = [500.0, 0.0]

    # Gradient slope for elution
    elution_duration = 5
    gradient_slope = [(c_elute_end[0] - c_elute_start[0]) / elution_duration,
                        (c_elute_end[1] - c_elute_start[1]) / elution_duration]

    # Polynomial coefficients for gradient: [const, linear, quad, cubic]
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
            h5_path = Path(tmpdir) / "test_result.h5"
            
            result = runner.run(process, h5_path=h5_path)
            
            assert result.success
            assert h5_path.exists(), "H5 file was not created"
            assert h5_path.stat().st_size > 0, "H5 file is empty"
    
    def test_analyzer_full_export(self):
        """Test full export workflow with real simulation."""
        from cadet_simplified.simulation import SimulationRunner
        from cadet_simplified.results import ResultsAnalyzer
        from cadet_simplified.operation_modes import ExperimentConfig, ColumnBindingConfig, ComponentDefinition
        
        process = create_simple_lwe_process()
        runner = SimulationRunner()
        result = runner.run(process)
        
        assert result.success, f"Simulation failed: {result.errors}"
        
        # Create mock configs
        experiment_config = ExperimentConfig(
            name="simple_lwe_test",
            parameters={
                "flow_rate_mL_min": 0.1,
                "load_cv": 2.0,
                "wash_cv": 2.0,
                "elution_cv": 4.0,
                "load_salt_mM": 50.0,
                "gradient_end_mM": 500.0,
            },
            components=[
                ComponentDefinition(name="Salt", is_salt=True),
                ComponentDefinition(name="Protein", is_salt=False),
            ],
        )
        
        column_binding = ColumnBindingConfig(
            column_model="LumpedRateModelWithoutPores",
            binding_model="Langmuir",
            column_parameters={
                "length": 0.01,
                "diameter": 0.01,
                "total_porosity": 0.7,
                "axial_dispersion": 1e-7,
            },
            binding_parameters={
                "is_kinetic": False,
            },
            component_column_parameters={},
            component_binding_parameters={
                "adsorption_rate": [0.0, 0.1],
                "desorption_rate": [0.0, 0.01],
                "capacity": [0.0, 100.0],
            },
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ResultsAnalyzer(
                base_dir=tmpdir,
                simulator=runner.simulator,
                n_interpolation_points=250,
            )
            
            output_path = analyzer.export(
                results=[result],
                experiment_configs=[experiment_config],
                column_binding=column_binding,
                name="integration_test",
            )
            
            # Check output folder exists
            assert output_path.exists()
            assert output_path.is_dir()
            
            # Check Excel file
            excel_path = output_path / "results.xlsx"
            assert excel_path.exists()
            
            # Verify Excel content
            xl = pd.ExcelFile(excel_path)
            assert "Parameters" in xl.sheet_names
            assert "simple_lwe_test" in xl.sheet_names
            
            # Check parameters sheet
            params_df = pd.read_excel(excel_path, sheet_name="Parameters")
            assert len(params_df) == 1
            assert params_df.iloc[0]["experiment_name"] == "simple_lwe_test"
            assert params_df.iloc[0]["simulation_success"] == True
            
            # Check chromatogram sheet
            chrom_df = pd.read_excel(excel_path, sheet_name="simple_lwe_test")
            assert len(chrom_df) == 250  # n_interpolation_points
            assert "time_s" in chrom_df.columns
            assert "c_Salt_mM" in chrom_df.columns
            assert "c_Protein_mM" in chrom_df.columns
            
            # Check H5 config file
            h5_files = list(output_path.glob("*.h5"))
            assert len(h5_files) == 1
            assert h5_files[0].name == "simple_lwe_test_config.h5"
    
    def test_analyzer_export_with_full_h5(self):
        """Test export when full H5 (with results) is preserved."""
        from cadet_simplified.simulation import SimulationRunner
        from cadet_simplified.results import ResultsAnalyzer
        from cadet_simplified.operation_modes import ExperimentConfig, ColumnBindingConfig, ComponentDefinition
        
        process = create_simple_lwe_process()
        runner = SimulationRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Run with H5 preservation
            output_path = tmpdir / "output"
            output_path.mkdir()
            
            h5_path = output_path / f"{process.name}.h5"
            result = runner.run(process, h5_path=h5_path)
            
            assert result.success
            assert h5_path.exists()
            
            # Create configs
            experiment_config = ExperimentConfig(
                name="simple_lwe_test",
                parameters={"flow_rate_mL_min": 0.1},
                components=[
                    ComponentDefinition(name="Salt", is_salt=True),
                    ComponentDefinition(name="Protein", is_salt=False),
                ],
            )
            
            column_binding = ColumnBindingConfig(
                column_model="LumpedRateModelWithoutPores",
                binding_model="Langmuir",
                column_parameters={"length": 0.01},
                binding_parameters={},
                component_column_parameters={},
                component_binding_parameters={},
            )
            
            analyzer = ResultsAnalyzer(
                base_dir=tmpdir,
                simulator=runner.simulator,
            )
            
            # Export using existing output_path (where H5 already exists)
            final_path = analyzer.export(
                results=[result],
                experiment_configs=[experiment_config],
                column_binding=column_binding,
                output_path=output_path,
            )
            
            # Should have full H5 (not _config.h5)
            h5_files = list(final_path.glob("*.h5"))
            assert len(h5_files) == 1
            # The full H5 should be kept, not replaced with config-only
            assert "simple_lwe_test.h5" in [f.name for f in h5_files]
            
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    process = create_simple_process()
    create_simple_configs()
    test = TestIntegrationSimulation()
    test.test_simple_simulation_runs()
    print("Done")