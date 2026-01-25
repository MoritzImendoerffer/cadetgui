from pathlib import Path
from cadet_simplified.operation_modes import (
    OPERATION_MODES,
    SUPPORTED_COLUMN_MODELS,
    SUPPORTED_BINDING_MODELS,
    get_operation_mode,
)
from cadet_simplified.excel import ExcelTemplateGenerator, ExcelParser
from cadet_simplified.storage import FileResultsStorage
from cadet_simplified.results import ResultsExporter  # For standalone Excel exports
from cadet_simplified.simulation import SimulationRunner

print(100*"=")
print("SUPPORTED COLUMN MODELS")
print(SUPPORTED_COLUMN_MODELS)
print(100*"-")
print("SUPPORTED BINDING MODELS")
print(SUPPORTED_BINDING_MODELS)
print(100*"-")
print("SUPPORTED OPERATION MODES")
print(OPERATION_MODES)
print(100*"=")

col = "LumpedRateModelWithoutPores"
bim = "StericMassAction"
opm = "LWE_concentration_based"

n_components = 3
temp_gen = ExcelTemplateGenerator(column_model=col, binding_model=bim, operation_mode=opm, n_components=n_components)
save_path = Path("~").expanduser()
excel_template_file = save_path.joinpath("test.xlsx")
temp_gen.save(excel_template_file)

load_path = Path("~").joinpath("test123_filled.xlsx").expanduser()
parser = ExcelParser()
result = parser.parse(load_path)

operation_mode = get_operation_mode(opm)
process_list = []
for exp in result.experiments:
    process = operation_mode.create_process(exp, result.column_binding)
    process_list.append(process)

# check if config is correct
[p.check_config() for p in process_list]

runner = SimulationRunner()

# Option A: Full storage (pickle + parquet chromatograms + H5)
storage = FileResultsStorage(save_path.joinpath("cadet_output"))
results = runner.run_batch(process_list, n_cores=3)
set_id = storage.save_experiment_set(
    name="my_study",
    operation_mode=opm,
    experiments=result.experiments,
    column_binding=result.column_binding,
    results=results,
)
print(f"Saved to storage with ID: {set_id}")

# Later, load for analysis:
# loaded = storage.load_results_by_selection([(set_id, exp.name) for exp in result.experiments])


# Option B: Just Excel export (no pickle/storage)
exporter = ResultsExporter(n_interpolation_points=500)
results = runner.run_batch(process_list, n_cores=3)
excel_path = exporter.export_simulation_results(
    results=results,
    experiment_configs=result.experiments,
    column_binding=result.column_binding,
    output_path=save_path.joinpath("cadet_output", "my_study_results.xlsx"),
)
print(f"Excel exported to: {excel_path}")