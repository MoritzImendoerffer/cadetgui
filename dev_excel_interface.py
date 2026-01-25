from pathlib import Path
from cadet_simplified.operation_modes import (
    OPERATION_MODES,
    SUPPORTED_COLUMN_MODELS,
    SUPPORTED_BINDING_MODELS,
    get_operation_mode,
)
from cadet_simplified.excel import ExcelTemplateGenerator, ExcelParser
from cadet_simplified.results import ResultsAnalyzer
from cadet_simplified.simulation import SimulationRunner
from CADETProcess.simulator import Cadet

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
col = "LumpedRateModelWithPores"
bim = "StericMassAction"
opm = "LWE_concentration_based"

col = "LumpedRateModelWithoutPores"

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
analyzer = ResultsAnalyzer(
    base_dir=save_path.joinpath("cadet_output"),
    simulator=Cadet(),
    n_interpolation_points=500,
    save_pickle=True,
)

# Option A: Quick run (config H5 only)
results = runner.run_batch(process_list, n_cores=3)
output_path = analyzer.export(results, result.experiments, result.column_binding, name="my_study")

# Option B: With full H5 preservation includint the results
output_path = analyzer.get_output_path("my_study")
results = runner.run_batch(process_list, h5_dir=output_path, n_cores=3)
analyzer.export(results, result.experiments, result.column_binding, output_path=output_path)