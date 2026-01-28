from pathlib import Path

from cadet_simplified import (
    list_column_models,
    list_binding_models,
    list_operation_modes,
    get_operation_mode,
    ExcelTemplateGenerator,
    parse_excel,
    plotting,
    SimulationRunner,
    # Storage
    FileStorage,
)
print(100 * "=")
print("SUPPORTED COLUMN MODELS")
print(list_column_models())
print(100 * "-")
print("SUPPORTED BINDING MODELS")
print(list_binding_models())
print(100 * "-")
print("SUPPORTED OPERATION MODES")
print(list_operation_modes())
print(100 * "=")

col = "LumpedRateModelWithoutPores"
bim = "StericMassAction"
opm = "LWE_concentration_based"
n_components = 3

template_generator = ExcelTemplateGenerator(
    operation_mode=opm,
    column_model=col,
    binding_model=bim,
    n_components=n_components,
    component_names=["Salt", "Product", "Impurity1"],
)

save_path = Path("~").expanduser()
excel_template_file = save_path / "test_template.xlsx"
template_generator.save(excel_template_file)
print(f"Template saved to: {excel_template_file}")


filled_template_path = save_path / "test_template_filled.xlsx"


result = parse_excel(filled_template_path)


operation_mode = get_operation_mode(opm)
process_list = []

for exp in result.experiments:
    process = operation_mode.create_process(exp, result.column_binding)
    process_list.append(process)
    

    print(process.check_config())


runner = SimulationRunner()

print("\nRunning simulations...")
results = runner.run_batch(
    process_list,
    stop_on_error=False,
    progress_callback=lambda current, total, res: print(
        f"  [{current}/{total}] {res.experiment_name}: "
        f"{'Success' if res.success else 'Failed'} ({res.runtime_seconds:.2f}s)"
    ),
)


storage_dir = save_path / "cadet_experiments"
storage = FileStorage(storage_dir)

set_id = storage.save_experiment_set(
    name="my_study",
    operation_mode=opm,
    experiments=result.experiments,
    column_binding=result.column_binding,
    results=results,
    description="Test simulation run",
)

# List what's stored
df = storage.list_experiments(limit=10)

# Load specific experiments
loaded = storage.load_results_by_selection(
    selections=[(set_id, exp.name) for exp in result.experiments],
    include_chromatogram=True,
)

print(f"\nLoaded {len(loaded)} experiment(s) for analysis")

p = plotting.plot_chromatogram(results[0])
