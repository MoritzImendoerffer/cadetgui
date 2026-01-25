from pathlib import Path
from cadet_simplified.operation_modes import (
    OPERATION_MODES,
    SUPPORTED_COLUMN_MODELS,
    SUPPORTED_BINDING_MODELS,
    get_operation_mode,
)
from cadet_simplified.excel import ExcelTemplateGenerator, ExcelParser, ParseResult
from cadet_simplified.storage import ExperimentStore, ExperimentSet
from cadet_simplified.simulation import SimulationRunner, ValidationResult

col = "LumpedRateModelWithPores"
bim = "StericMassAction"
opm = "LWE_concentration_based"

col = "LumpedRateModelWithoutPores"

n_components = 3
temp_gen = ExcelTemplateGenerator(column_model=col, binding_model=bim, operation_mode=opm, n_components=n_components)
save_path = Path("~").expanduser()
excel_file = save_path.joinpath("test.xlsx")
temp_gen.save(save_path)

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
result = runner.run(process_list[0])

results = runner.run_batch(process_list, n_cores=3)


print("Done")