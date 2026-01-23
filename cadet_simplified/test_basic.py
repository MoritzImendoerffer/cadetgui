"""Test script for cadet_simplified package.

Run with: python -m cadet_simplified.test_basic
"""

import sys
from pathlib import Path

# Add parent to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_template_generation():
    """Test Excel template generation."""
    print("=" * 60)
    print("Testing Excel Template Generation")
    print("=" * 60)
    
    from cadet_simplified import get_lwe_mode, ExcelTemplateGenerator
    
    mode = get_lwe_mode()
    print(f"Operation mode: {mode.name}")
    print(f"Description: {mode.description}")
    
    # Get experiment parameters
    print("\nExperiment parameters:")
    for param in mode.get_experiment_parameters():
        print(f"  - {param.name}: {param.display_name} [{param.unit}]")
    
    print("\nPer-component experiment parameters:")
    for param in mode.get_component_experiment_parameters():
        print(f"  - {param.name}: {param.display_name} [{param.unit}]")
    
    # Generate template
    generator = ExcelTemplateGenerator(
        operation_mode=mode,
        column_model="LumpedRateModelWithPores",
        binding_model="StericMassAction",
        n_components=4,
        component_names=["Salt", "Product", "Impurity1", "Impurity2"],
    )
    
    sheets = generator.generate()
    
    print("\nGenerated sheets:")
    for name, df in sheets.items():
        print(f"\n  {name}:")
        print(f"    Columns: {list(df.columns)[:5]}...")
        print(f"    Rows: {len(df)}")
    
    # Save template
    output_path = Path("/tmp/test_template.xlsx")
    generator.save(str(output_path))
    print(f"\nTemplate saved to: {output_path}")
    
    return True


def test_excel_parsing():
    """Test Excel parsing."""
    print("\n" + "=" * 60)
    print("Testing Excel Parsing")
    print("=" * 60)
    
    from cadet_simplified import parse_excel
    
    # Parse the template we just generated
    result = parse_excel("/tmp/test_template.xlsx")
    
    print(f"Parse success: {result.success}")
    print(f"Experiments: {len(result.experiments)}")
    
    if result.experiments:
        exp = result.experiments[0]
        print(f"\nFirst experiment: {exp.name}")
        print(f"  Components: {[c.name for c in exp.components]}")
        print(f"  Parameters: {list(exp.parameters.keys())[:5]}...")
    
    if result.column_binding:
        cb = result.column_binding
        print(f"\nColumn model: {cb.column_model}")
        print(f"Binding model: {cb.binding_model}")
        print(f"Column params: {list(cb.column_parameters.keys())}")
    
    if result.errors:
        print(f"\nErrors: {result.errors}")
    if result.warnings:
        print(f"\nWarnings: {result.warnings}")
    
    return result.success


def test_process_creation():
    """Test process creation (requires CADET-Process)."""
    print("\n" + "=" * 60)
    print("Testing Process Creation")
    print("=" * 60)
    
    try:
        from CADETProcess.processModel import ComponentSystem
        print("CADET-Process is available")
    except ImportError:
        print("CADET-Process not installed - skipping process creation test")
        return True
    
    from cadet_simplified import get_lwe_mode, parse_excel
    
    mode = get_lwe_mode()
    result = parse_excel("/tmp/test_template.xlsx")
    
    if not result.success or not result.experiments:
        print("No experiments to test")
        return False
    
    exp = result.experiments[0]
    
    # We need to fill in required parameters
    # The template has defaults but some may be missing
    # Let's add minimal required values
    if not exp.parameters.get("flow_rate_cv_min"):
        exp.parameters["flow_rate_cv_min"] = 1.0
    if not exp.parameters.get("load_cv"):
        exp.parameters["load_cv"] = 5.0
    
    # Fill in column parameters if missing
    cb = result.column_binding
    if not cb.column_parameters.get("length"):
        cb.column_parameters["length"] = 10.0  # cm
    if not cb.column_parameters.get("diameter"):
        cb.column_parameters["diameter"] = 1.0  # cm
    if not cb.column_parameters.get("bed_porosity"):
        cb.column_parameters["bed_porosity"] = 0.37
    if not cb.column_parameters.get("particle_porosity"):
        cb.column_parameters["particle_porosity"] = 0.33
    if not cb.column_parameters.get("particle_radius"):
        cb.column_parameters["particle_radius"] = 34.0  # µm
    if not cb.column_parameters.get("axial_dispersion"):
        cb.column_parameters["axial_dispersion"] = 1e-7
    
    # Fill binding parameters
    if not cb.binding_parameters.get("capacity"):
        cb.binding_parameters["capacity"] = 1200.0
    if "is_kinetic" not in cb.binding_parameters:
        cb.binding_parameters["is_kinetic"] = True
    
    # Fill per-component parameters
    n_comp = len(exp.components)
    if not cb.component_column_parameters.get("film_diffusion"):
        cb.component_column_parameters["film_diffusion"] = [1e-4] * n_comp
    if not cb.component_column_parameters.get("pore_diffusion"):
        cb.component_column_parameters["pore_diffusion"] = [1e-9] * n_comp
    
    # SMA binding parameters
    if not cb.component_binding_parameters.get("adsorption_rate"):
        cb.component_binding_parameters["adsorption_rate"] = [0.0, 35.5, 7.7, 7.7]
    if not cb.component_binding_parameters.get("desorption_rate"):
        cb.component_binding_parameters["desorption_rate"] = [0.0, 1000.0, 1000.0, 1000.0]
    if not cb.component_binding_parameters.get("characteristic_charge"):
        cb.component_binding_parameters["characteristic_charge"] = [0.0, 4.7, 5.29, 5.29]
    if not cb.component_binding_parameters.get("steric_factor"):
        cb.component_binding_parameters["steric_factor"] = [0.0, 11.83, 10.6, 10.6]
    
    try:
        process = mode.create_process(exp, cb)
        print(f"Process created: {process.name}")
        print(f"Cycle time: {process.cycle_time:.1f} s")
        print(f"Components: {[c.name for c in process.component_system.components]}")
        
        # Try validation
        is_valid = process.check_config()
        print(f"Configuration valid: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"Error creating process: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_storage():
    """Test experiment storage."""
    print("\n" + "=" * 60)
    print("Testing Experiment Storage")
    print("=" * 60)
    
    from cadet_simplified import ExperimentStore, parse_excel
    import tempfile
    
    # Create temporary storage
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExperimentStore(tmpdir)
        
        # Parse template
        result = parse_excel("/tmp/test_template.xlsx")
        
        if not result.success:
            print("Parse failed, skipping storage test")
            return False
        
        # Save experiment set
        exp_set = store.save_from_parse_result(
            experiments=result.experiments,
            column_binding=result.column_binding,
            name="Test Experiment Set",
            operation_mode="LWE_concentration_based",
            description="Testing storage",
        )
        
        print(f"Saved experiment set: {exp_set.id}")
        print(f"  Name: {exp_set.name}")
        print(f"  Experiments: {len(exp_set.experiments)}")
        
        # List all
        all_sets = store.list_all()
        print(f"\nAll experiment sets: {len(all_sets)}")
        
        # Load back
        loaded = store.load(exp_set.id)
        print(f"\nLoaded back: {loaded.name}")
        print(f"  Experiments: {len(loaded.experiments)}")
        
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CADET Simplified - Basic Tests")
    print("=" * 60 + "\n")
    
    results = {}
    
    results["template_generation"] = test_template_generation()
    results["excel_parsing"] = test_excel_parsing()
    results["process_creation"] = test_process_creation()
    results["storage"] = test_storage()
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
