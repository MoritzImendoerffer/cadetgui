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
from cadet_simplified.analysis.plotting import overlay_plot, show_plot
from cadet_simplified.analysis import SimpleChromatogramAnalysis
import holoviews as hv
import numpy as np
import pandas as pd
load_path = Path("/home/moritz/Downloads").joinpath("example_mara.xlsx").expanduser()
save_path = Path("/home/moritz/Downloads").joinpath("example_mara_results")
parser = ExcelParser()
result = parser.parse(load_path)

opm = "LWE_concentration_based"
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

# Run simulations with H5 files in the pending directory
# This keeps temporary files separate from finalized experiment sets
results = runner.run_batch(process_list, h5_dir=storage.get_pending_dir(), n_cores=3)

def overlay_plot(results, show_events = True, n_points=1000, width=1200, height=1000):
    """Create hvplot overlay of chromatograms."""
    
    plots = []
    
    for result in results:
        if result.cadet_result is None:
            continue
        chrom = result.cadet_result.chromatograms
        assert len(chrom) == 1
        chrom = chrom[0]
        time = np.linspace(chrom.time[0], chrom.time[-1], n_points)
        conc = chrom.solution_interpolated(time)
        names = [item + "_mM" for item in chrom.component_system.names]
        df = pd.DataFrame(conc, columns=names)
        df["time_min"] = time / 60
        # Create label for legend
        label_prefix = result.experiment_name
        
        # Plot each component
        for comp in  names:
            label = f"{label_prefix} - {comp}"
            plot = df.hvplot.line(
                x='time_min',
                y=comp,
                label=label,
            )
            plots.append(plot)
        
        if show_events:
            events = result.cadet_result.process.events
            event_times = [item.time/60 for item in events]
            event_names = [item.name for item in events]
            text = [hv.VLine(t).opts(color="gray")*hv.Text(t, 10, n) for t,n in zip(event_times, event_names) ]
            plots.extend(text)            
    if not plots:
        raise ValueError("No chromatogram data available for plotting")
    
    # Overlay all plots
    overlay = plots[0]
    for p in plots[1:]:
        overlay = overlay * p
    

    # Configure plot options
    overlay = overlay.opts(
        xlabel='Time (min)',
        ylabel='Concentration (mM)',
        title='Chromatogram Overlay',
        legend_position='right',
        width=width,
        height=height,
        tools=['hover', 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'],
        active_tools=['wheel_zoom'],
    )
    
    return overlay


p = overlay_plot(results, width=1800, height=1000)
show_plot(p)


analyzer = SimpleChromatogramAnalysis()


# Save experiment set - this moves H5 files from _pending to the final location
set_id = storage.save_experiment_set(
    name="my_study",
    operation_mode=opm,
    experiments=result.experiments,
    column_binding=result.column_binding,
    results=results,
)
print(f"Saved to storage with ID: {set_id}")