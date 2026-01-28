import numpy as np
import pandas as pd
import hvplot.pandas
import holoviews as hv
hv.extension('bokeh')
from bokeh.plotting import show

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
            events = result.process.events
            event_times = [item.time/60 for item in events]
            event_names = [item.name for item in events]
            line = hv.VLines(event_times)
            text = [hv.Text(t, 10, n) for t,n in zip(event_times, event_names) ]
            for t in text:
                line = line*t
            plots.append(t)            
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

def show_plot(hvplot):
    return show(hv.render(hvplot))