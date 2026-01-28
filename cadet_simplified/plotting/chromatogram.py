"""Chromatogram plotting functions.

Provides reusable plotting functions that work in both Jupyter notebooks
and Panel applications. All functions return HoloViews/hvplot objects.

Three levels of usage:

1. From simulation result (with interpolation):
   >>> plot = plot_chromatogram(result, n_points=2000)

2. From cached DataFrame:
   >>> plot = plot_chromatogram_from_df(df, title="My Experiment")

3. Overlay multiple experiments:
   >>> plot = plot_chromatogram_overlay(results, labels=["Exp 1", "Exp 2"])

The interpolation is separated so it can be cached:
   >>> df = interpolate_chromatogram(result)  # Save this
   >>> plot = plot_chromatogram_from_df(df)   # Fast reuse
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import holoviews as hv

hv.extension("bokeh")

if TYPE_CHECKING:
    from ..simulation.runner import SimulationResultWrapper


# =============================================================================
# Interpolation utilities
# =============================================================================

def interpolate_chromatogram(
    result: "SimulationResultWrapper",
    n_points: int = 2000,
    outlet_name: str | None = None,
    time_unit: str = "seconds",
) -> pd.DataFrame:
    """Extract and interpolate chromatogram from simulation result.
    
    Parameters
    ----------
    result : SimulationResultWrapper
        Simulation result with cadet_result
    n_points : int, default=2000
        Number of interpolation points
    outlet_name : str, optional
        Name of outlet unit. If None, uses first product outlet.
    time_unit : str, default="seconds"
        Time unit for output: "seconds" or "minutes"
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, <component_names>...
        
    Raises
    ------
    ValueError
        If result has no cadet_result or no outlet
        
    Example
    -------
    >>> df = interpolate_chromatogram(result, n_points=2000)
    >>> df.to_parquet("chromatogram.parquet")  # Cache for later
    """
    if result.cadet_result is None:
        raise ValueError(f"No cadet_result in {result.experiment_name}")
    
    cadet_result = result.cadet_result
    process = cadet_result.process
    
    # Find outlet
    if outlet_name is not None:
        outlet = process.flow_sheet[outlet_name]
    else:
        product_outlets = process.flow_sheet.product_outlets
        if not product_outlets:
            raise ValueError(f"No product outlet found for {result.experiment_name}")
        outlet = product_outlets[0]
    
    # Get solution
    outlet_solution = cadet_result.solution[outlet.name]
    
    # Create interpolation time points
    time_complete = cadet_result.time_complete
    time_interp = np.linspace(
        float(time_complete.min()),
        float(time_complete.max()),
        n_points,
    )
    
    # Interpolate
    interp_func = outlet_solution.outlet.solution_interpolated
    solution_interp = interp_func(time_interp)
    
    # Convert time if needed
    if time_unit == "minutes":
        time_output = time_interp / 60.0
    else:
        time_output = time_interp
    
    # Build DataFrame
    data = {"time": time_output}
    for i, comp in enumerate(process.component_system.components):
        comp_name = comp.name if hasattr(comp, "name") else f"Component_{i}"
        data[comp_name] = solution_interp[:, i]
    
    return pd.DataFrame(data)


def get_component_names(result: "SimulationResultWrapper") -> list[str]:
    """Get component names from a simulation result.
    
    Parameters
    ----------
    result : SimulationResultWrapper
        Simulation result with cadet_result
        
    Returns
    -------
    list[str]
        Component names
    """
    if result.cadet_result is None:
        return []
    
    process = result.cadet_result.process
    names = []
    for comp in process.component_system.components:
        name = comp.name if hasattr(comp, "name") else f"Component_{len(names)}"
        names.append(name)
    return names


# =============================================================================
# Plotting from DataFrame (cached data)
# =============================================================================

def plot_chromatogram_from_df(
    df: pd.DataFrame,
    title: str = "Chromatogram",
    time_unit: str = "auto",
    ylabel: str = "Concentration (mM)",
    width: int = 800,
    height: int = 400,
    show_legend: bool = True,
) -> hv.Overlay:
    """Plot chromatogram from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'time' column and component columns
    title : str, default="Chromatogram"
        Plot title
    time_unit : str, default="auto"
        X-axis label: "auto" (detect from data), "seconds", "minutes"
    ylabel : str, default="Concentration (mM)"
        Y-axis label
    width : int, default=800
        Plot width in pixels
    height : int, default=400
        Plot height in pixels
    show_legend : bool, default=True
        Whether to show legend
        
    Returns
    -------
    hv.Overlay
        HoloViews plot object
        
    Example
    -------
    >>> df = pd.read_parquet("chromatogram.parquet")
    >>> plot = plot_chromatogram_from_df(df, title="Gradient Elution")
    >>> plot  # Display in notebook
    """
    import hvplot.pandas  # noqa: F401
    
    # Determine time unit from data
    if time_unit == "auto":
        max_time = df["time"].max()
        if max_time > 300:  # Likely seconds
            xlabel = "Time (s)"
        else:
            xlabel = "Time (min)"
    elif time_unit == "minutes":
        xlabel = "Time (min)"
    else:
        xlabel = "Time (s)"
    
    # Get component columns (everything except 'time')
    component_cols = [c for c in df.columns if c != "time"]
    
    if not component_cols:
        raise ValueError("No component columns found in DataFrame")
    
    # Create individual plots
    plots = []
    for comp in component_cols:
        p = df.hvplot.line(
            x="time",
            y=comp,
            label=comp,
        )
        plots.append(p)
    
    # Overlay
    if len(plots) == 1:
        overlay = plots[0]
    else:
        overlay = plots[0]
        for p in plots[1:]:
            overlay = overlay * p
    
    # Configure options
    legend_position = "right" if show_legend else None
    
    overlay = overlay.opts(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        width=width,
        height=height,
        legend_position=legend_position,
        show_legend=show_legend,
        tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset", "save"],
        active_tools=["wheel_zoom"],
    )
    
    return overlay


def plot_chromatogram_overlay_from_df(
    chromatograms: list[tuple[str, pd.DataFrame]],
    title: str = "Chromatogram Overlay",
    time_unit: str = "auto",
    ylabel: str = "Concentration (mM)",
    width: int = 900,
    height: int = 450,
    component_filter: list[str] | None = None,
) -> hv.Overlay:
    """Overlay multiple chromatograms from DataFrames.
    
    Parameters
    ----------
    chromatograms : list[tuple[str, pd.DataFrame]]
        List of (label, dataframe) pairs
    title : str, default="Chromatogram Overlay"
        Plot title
    time_unit : str, default="auto"
        X-axis label
    ylabel : str, default="Concentration (mM)"
        Y-axis label
    width : int, default=900
        Plot width
    height : int, default=450
        Plot height
    component_filter : list[str], optional
        Only plot these components. If None, plot all.
        
    Returns
    -------
    hv.Overlay
        HoloViews plot object
        
    Example
    -------
    >>> chromatograms = [
    ...     ("Experiment 1", df1),
    ...     ("Experiment 2", df2),
    ... ]
    >>> plot = plot_chromatogram_overlay_from_df(chromatograms)
    """
    import hvplot.pandas  # noqa: F401
    
    if not chromatograms:
        raise ValueError("No chromatograms provided")
    
    # Determine time unit from first DataFrame
    if time_unit == "auto":
        max_time = chromatograms[0][1]["time"].max()
        xlabel = "Time (s)" if max_time > 300 else "Time (min)"
    elif time_unit == "minutes":
        xlabel = "Time (min)"
    else:
        xlabel = "Time (s)"
    
    # Create plots
    plots = []
    
    for label, df in chromatograms:
        # Get component columns
        component_cols = [c for c in df.columns if c != "time"]
        
        if component_filter is not None:
            component_cols = [c for c in component_cols if c in component_filter]
        
        for comp in component_cols:
            # Create label: "Experiment / Component" or just "Experiment" if single component
            if len(component_cols) > 1 or len(chromatograms) > 1:
                full_label = f"{label} - {comp}"
            else:
                full_label = label
            
            p = df.hvplot.line(
                x="time",
                y=comp,
                label=full_label,
            )
            plots.append(p)
    
    if not plots:
        raise ValueError("No data to plot")
    
    # Overlay all plots
    overlay = plots[0]
    for p in plots[1:]:
        overlay = overlay * p
    
    overlay = overlay.opts(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        width=width,
        height=height,
        legend_position="right",
        tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset", "save"],
        active_tools=["wheel_zoom"],
    )
    
    return overlay


# =============================================================================
# Plotting from SimulationResultWrapper (with interpolation)
# =============================================================================

def plot_chromatogram(
    result: "SimulationResultWrapper",
    n_points: int = 2000,
    outlet_name: str | None = None,
    title: str | None = None,
    time_unit: str = "minutes",
    **kwargs,
) -> hv.Overlay:
    """Plot chromatogram from simulation result.
    
    Interpolates the result and creates a plot. For repeated plotting
    of the same result, use interpolate_chromatogram() to cache the data,
    then plot_chromatogram_from_df().
    
    Parameters
    ----------
    result : SimulationResultWrapper
        Simulation result with cadet_result
    n_points : int, default=2000
        Number of interpolation points
    outlet_name : str, optional
        Name of outlet unit
    title : str, optional
        Plot title. If None, uses experiment name.
    time_unit : str, default="minutes"
        Time unit: "seconds" or "minutes"
    **kwargs
        Additional arguments passed to plot_chromatogram_from_df
        
    Returns
    -------
    hv.Overlay
        HoloViews plot object
        
    Example
    -------
    >>> result = runner.run(process)
    >>> plot = plot_chromatogram(result)
    >>> plot  # Display in notebook
    """
    # Interpolate
    df = interpolate_chromatogram(
        result,
        n_points=n_points,
        outlet_name=outlet_name,
        time_unit=time_unit,
    )
    
    # Set title
    if title is None:
        title = result.experiment_name
    
    # Plot
    return plot_chromatogram_from_df(
        df,
        title=title,
        time_unit=time_unit,
        **kwargs,
    )


def plot_chromatogram_overlay(
    results: list["SimulationResultWrapper"],
    n_points: int = 2000,
    labels: list[str] | None = None,
    title: str = "Chromatogram Overlay",
    time_unit: str = "minutes",
    component_filter: list[str] | None = None,
    **kwargs,
) -> hv.Overlay:
    """Overlay multiple chromatograms from simulation results.
    
    Parameters
    ----------
    results : list[SimulationResultWrapper]
        List of simulation results
    n_points : int, default=2000
        Number of interpolation points
    labels : list[str], optional
        Labels for each result. If None, uses experiment names.
    title : str, default="Chromatogram Overlay"
        Plot title
    time_unit : str, default="minutes"
        Time unit
    component_filter : list[str], optional
        Only plot these components
    **kwargs
        Additional arguments passed to plot_chromatogram_overlay_from_df
        
    Returns
    -------
    hv.Overlay
        HoloViews plot object
    """
    if labels is None:
        labels = [r.experiment_name for r in results]
    
    if len(labels) != len(results):
        raise ValueError("Number of labels must match number of results")
    
    # Interpolate all
    chromatograms = []
    for label, result in zip(labels, results):
        if result.cadet_result is None:
            continue
        df = interpolate_chromatogram(
            result,
            n_points=n_points,
            time_unit=time_unit,
        )
        chromatograms.append((label, df))
    
    if not chromatograms:
        raise ValueError("No valid results to plot")
    
    return plot_chromatogram_overlay_from_df(
        chromatograms,
        title=title,
        time_unit=time_unit,
        component_filter=component_filter,
        **kwargs,
    )
