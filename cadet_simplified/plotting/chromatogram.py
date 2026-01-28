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

from typing import TYPE_CHECKING
import math

import numpy as np
import pandas as pd
import holoviews as hv

hv.extension("bokeh")

if TYPE_CHECKING:
    from ..simulation.runner import SimulationResultWrapper
    from CADETProcess.processModel import Process


# =============================================================================
# Helper Functions for Unit Conversion
# =============================================================================

def calculate_column_volume_mL(column_parameters: dict) -> float:
    """Calculate column volume in mL from column parameters.
    
    Parameters
    ----------
    column_parameters : dict
        Dictionary containing 'length' (m) and 'diameter' (m)
        
    Returns
    -------
    float
        Column volume in mL
        
    Raises
    ------
    ValueError
        If length or diameter not found in parameters
    """
    length = column_parameters.get("length")
    diameter = column_parameters.get("diameter")
    
    if length is None or diameter is None:
        raise ValueError(
            "Column parameters must contain 'length' and 'diameter'. "
            f"Got: {list(column_parameters.keys())}"
        )
    
    # Volume = π * r² * L (in m³)
    radius = diameter / 2.0
    volume_m3 = math.pi * radius ** 2 * length
    
    # Convert to mL (1 m³ = 1e6 mL)
    return volume_m3 * 1e6


def time_to_cv(
    time_seconds: np.ndarray,
    flow_rate_mL_min: float,
    column_volume_mL: float,
) -> np.ndarray:
    """Convert time array from seconds to column volumes.
    
    Parameters
    ----------
    time_seconds : np.ndarray
        Time array in seconds
    flow_rate_mL_min : float
        Flow rate in mL/min
    column_volume_mL : float
        Column volume in mL
        
    Returns
    -------
    np.ndarray
        Time array in column volumes (CV)
    """
    # time (s) → time (min) → volume (mL) → CV
    time_minutes = time_seconds / 60.0
    volume_mL = time_minutes * flow_rate_mL_min
    return volume_mL / column_volume_mL


def get_column_volume_from_process(process: "Process") -> float:
    """Extract column volume in mL from a Process object.
    
    Parameters
    ----------
    process : Process
        CADET-Process Process object
        
    Returns
    -------
    float
        Column volume in mL
    """
    # Find the column unit
    for unit in process.flow_sheet.units:
        if hasattr(unit, 'volume'):
            # volume property returns m³
            return unit.volume * 1e6
    
    raise ValueError("No column unit with 'volume' attribute found in process")


def get_flow_rate_from_process(process: "Process") -> float:
    """Extract flow rate in mL/min from a Process object.
    
    Parameters
    ----------
    process : Process
        CADET-Process Process object
        
    Returns
    -------
    float
        Flow rate in mL/min
    """
    # Get inlet and its flow rate (m³/s)
    inlet = process.flow_sheet.feed_inlets[0]
    flow_rate_m3_s = inlet.flow_rate
    
    # Convert to mL/min
    return flow_rate_m3_s * 1e6 * 60.0


# =============================================================================
# Interpolation Functions
# =============================================================================

def interpolate_chromatogram(
    result: "SimulationResultWrapper",
    n_points: int = 2000,
    outlet_name: str | None = None,
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
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time (in seconds), <component_names>...
        
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
    
    # Build DataFrame (time in seconds)
    data = {"time": time_interp}
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
# Inlet Profile Plotting (from Process events)
# =============================================================================

def get_inlet_profile_df(
    process: "Process",
    n_points: int = 1001,
    inlet_path: str = "flow_sheet.inlet.c",
) -> pd.DataFrame:
    """Extract inlet concentration profile from process events.
    
    Parameters
    ----------
    process : Process
        CADETProcess Process object with events defined
    n_points : int, default=1001
        Number of time points for interpolation
    inlet_path : str, default="flow_sheet.inlet.c"
        Parameter path for the inlet concentration
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time (in seconds), <component_names>...
        
    Raises
    ------
    ValueError
        If inlet parameter timeline not found
    """
    # Get the parameter timelines from the process
    parameter_timelines = process.parameter_timelines
    
    if inlet_path not in parameter_timelines:
        raise ValueError(
            f"Inlet path '{inlet_path}' not found in parameter_timelines. "
            f"Available: {list(parameter_timelines.keys())}"
        )
    
    timeline = parameter_timelines[inlet_path]
    
    # Generate time array
    time_s = np.linspace(0, process.cycle_time, n_points)
    
    # Get values from timeline
    values = timeline.value(time_s)
    
    # Ensure 2D array (n_points, n_components)
    values = np.atleast_2d(values)
    if values.shape[0] != n_points:
        values = values.T
    
    # Build DataFrame
    data = {"time": time_s}
    
    # Get component names from component system
    component_system = process.component_system
    for i, comp in enumerate(component_system.components):
        comp_name = comp.name if hasattr(comp, "name") else f"Component_{i}"
        if i < values.shape[1]:
            data[comp_name] = values[:, i]
    
    return pd.DataFrame(data)


def plot_inlet_profile(
    process: "Process",
    n_points: int = 1001,
    title: str = "Inlet Concentration Profile",
    normalized: bool = False,
    x_axis: str = "minutes",
    width: int = 800,
    height: int = 400,
    inlet_path: str = "flow_sheet.inlet.c",
) -> hv.Overlay:
    """Plot inlet concentration over time from process events.
    
    Uses the process's parameter_timelines to reconstruct the inlet
    concentration profile over time, as defined by the events.
    
    Parameters
    ----------
    process : Process
        CADETProcess Process object with events defined
    n_points : int, default=1001
        Number of time points for interpolation
    title : str, default="Inlet Concentration Profile"
        Plot title
    normalized : bool, default=False
        If True, normalize each component to its maximum value
    x_axis : str, default="minutes"
        X-axis unit: "minutes" or "cv" (column volumes)
    width : int, default=800
        Plot width in pixels
    height : int, default=400
        Plot height in pixels
    inlet_path : str, default="flow_sheet.inlet.c"
        Parameter path for the inlet concentration
        
    Returns
    -------
    hv.Overlay
        HoloViews plot object
        
    Example
    -------
    >>> from cadet_simplified.plotting import plot_inlet_profile
    >>> plot = plot_inlet_profile(process, x_axis="cv")
    >>> plot  # Display in notebook
    """
    import hvplot.pandas  # noqa: F401
    
    # Get the inlet profile DataFrame
    df = get_inlet_profile_df(process, n_points=n_points, inlet_path=inlet_path)
    
    # Convert x-axis
    if x_axis == "cv":
        column_volume_mL = get_column_volume_from_process(process)
        flow_rate_mL_min = get_flow_rate_from_process(process)
        df["time"] = time_to_cv(df["time"].values, flow_rate_mL_min, column_volume_mL)
        xlabel = "Column Volumes (CV)"
    else:
        # Convert time from seconds to minutes
        df["time"] = df["time"] / 60.0
        xlabel = "Time (min)"
    
    # Get component columns
    component_cols = [c for c in df.columns if c != "time"]
    
    # Normalize if requested
    if normalized:
        for comp in component_cols:
            max_val = df[comp].max()
            if max_val > 0:
                df[comp] = df[comp] / max_val
    
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
    
    # Set ylabel based on normalization
    ylabel = "Normalized Concentration (-)" if normalized else "Concentration (mM)"
    
    overlay = overlay.opts(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        width=width,
        height=height,
        legend_position="right",
        show_legend=True,
        tools=["hover", "pan", "wheel_zoom", "box_zoom", "reset", "save"],
        active_tools=["wheel_zoom"],
    )
    
    return overlay


# =============================================================================
# Chromatogram Plotting (from DataFrame)
# =============================================================================

def plot_chromatogram_from_df(
    df: pd.DataFrame,
    title: str = "Chromatogram",
    ylabel: str = "Concentration (mM)",
    width: int = 800,
    height: int = 400,
    show_legend: bool = True,
    normalized: bool = False,
    x_axis: str = "minutes",
    flow_rate_mL_min: float | None = None,
    column_volume_mL: float | None = None,
) -> hv.Overlay:
    """Plot chromatogram from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'time' column (in seconds) and component columns
    title : str, default="Chromatogram"
        Plot title
    ylabel : str, default="Concentration (mM)"
        Y-axis label
    width : int, default=800
        Plot width in pixels
    height : int, default=400
        Plot height in pixels
    show_legend : bool, default=True
        Whether to show legend
    normalized : bool, default=False
        If True, normalize each component to its maximum value
    x_axis : str, default="minutes"
        X-axis unit: "minutes" or "cv" (column volumes)
    flow_rate_mL_min : float, optional
        Flow rate in mL/min (required if x_axis="cv")
    column_volume_mL : float, optional
        Column volume in mL (required if x_axis="cv")
        
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

    _df = df.copy()
    
    # Get component columns (everything except 'time')
    component_cols = [c for c in _df.columns if "time" not in c.lower()]
    time_cols = [c for c in _df.columns if "time" in c.lower()]
    
    if not time_cols:
        raise ValueError("No time column found in DataFrame")
    if not component_cols:
        raise ValueError("No component columns found in DataFrame")
    
    time_col = time_cols[0]
    
    # Convert x-axis
    if x_axis == "cv":
        if flow_rate_mL_min is None or column_volume_mL is None:
            raise ValueError(
                "flow_rate_mL_min and column_volume_mL are required when x_axis='cv'"
            )
        _df[time_col] = time_to_cv(
            _df[time_col].values, flow_rate_mL_min, column_volume_mL
        )
        xlabel = "Column Volumes (CV)"
    else:
        # Convert time from seconds to minutes
        _df[time_col] = _df[time_col] / 60.0
        xlabel = "Time (min)"
    
    # Normalize if requested
    if normalized:
        for comp in component_cols:
            max_val = _df[comp].max()
            if max_val > 0:
                _df[comp] = _df[comp] / max_val
    
    # Create individual plots
    plots = []
    for comp in component_cols:
        p = _df.hvplot.line(
            x=time_col,
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
    
    legend_position = "right" if show_legend else None
    
    # Adjust ylabel for normalized plots
    if normalized:
        ylabel = "Normalized Concentration (-)"
    
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
    chromatograms: list[tuple[str, pd.DataFrame]] | list[tuple[str, pd.DataFrame, dict]],
    title: str = "Chromatogram Overlay",
    ylabel: str = "Concentration (mM)",
    width: int = 900,
    height: int = 450,
    component_filter: list[str] | None = None,
    normalized: bool = False,
    x_axis: str = "minutes",
) -> hv.Overlay:
    """Overlay multiple chromatograms from DataFrames.
    
    Parameters
    ----------
    chromatograms : list of tuples
        Either list of (label, dataframe) pairs, or
        list of (label, dataframe, conversion_params) where conversion_params
        is a dict with "flow_rate_mL_min" and "column_volume_mL" keys.
        Conversion params are required when x_axis="cv".
    title : str, default="Chromatogram Overlay"
        Plot title
    ylabel : str, default="Concentration (mM)"
        Y-axis label
    width : int, default=900
        Plot width
    height : int, default=450
        Plot height
    component_filter : list[str], optional
        Only plot these components. If None, plot all.
    normalized : bool, default=False
        If True, normalize each component to its maximum value
    x_axis : str, default="minutes"
        X-axis unit: "minutes" or "cv" (column volumes)
        
    Returns
    -------
    hv.Overlay
        HoloViews plot object
        
    Example
    -------
    >>> chromatograms = [
    ...     ("Experiment 1", df1, {"flow_rate_mL_min": 1.0, "column_volume_mL": 0.785}),
    ...     ("Experiment 2", df2, {"flow_rate_mL_min": 1.0, "column_volume_mL": 0.785}),
    ... ]
    >>> plot = plot_chromatogram_overlay_from_df(chromatograms, x_axis="cv")
    """
    import hvplot.pandas  # noqa: F401
    
    if not chromatograms:
        raise ValueError("No chromatograms provided")
    
    # Determine xlabel
    if x_axis == "cv":
        xlabel = "Column Volumes (CV)"
    else:
        xlabel = "Time (min)"
    
    # Create plots
    plots = []
    
    for item in chromatograms:
        # Unpack - support both 2-tuple and 3-tuple
        if len(item) == 2:
            label, df = item
            conv_params = None
        else:
            label, df, conv_params = item
        
        _df = df.copy()
        
        # Find time column
        time_cols = [c for c in _df.columns if "time" in c.lower()]
        if not time_cols:
            raise ValueError(f"No time column found in DataFrame for {label}")
        time_col = time_cols[0]
        
        # Convert x-axis
        if x_axis == "cv":
            if conv_params is None:
                raise ValueError(
                    f"Conversion params required for '{label}' when x_axis='cv'. "
                    "Provide (label, df, {'flow_rate_mL_min': ..., 'column_volume_mL': ...})"
                )
            flow_rate = conv_params.get("flow_rate_mL_min")
            col_vol = conv_params.get("column_volume_mL")
            if flow_rate is None or col_vol is None:
                raise ValueError(
                    f"Conversion params for '{label}' must contain "
                    "'flow_rate_mL_min' and 'column_volume_mL'"
                )
            _df[time_col] = time_to_cv(_df[time_col].values, flow_rate, col_vol)
        else:
            # Convert time from seconds to minutes
            _df[time_col] = _df[time_col] / 60.0
        
        # Get component columns
        component_cols = [c for c in _df.columns if "time" not in c.lower()]
        
        if component_filter is not None:
            component_cols = [c for c in component_cols if c in component_filter]
        
        # Normalize if requested
        if normalized:
            for comp in component_cols:
                max_val = _df[comp].max()
                if max_val > 0:
                    _df[comp] = _df[comp] / max_val
        
        for comp in component_cols:
            # Create label: "Experiment / Component" or just "Experiment" if single component
            if len(component_cols) > 1 or len(chromatograms) > 1:
                full_label = f"{label} - {comp}"
            else:
                full_label = label
            
            p = _df.hvplot.line(
                x=time_col,
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
    
    # Adjust ylabel for normalized plots
    if normalized:
        ylabel = "Normalized Concentration (-)"
    
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
# Chromatogram Plotting (from SimulationResultWrapper)
# =============================================================================

def plot_chromatogram(
    result: "SimulationResultWrapper",
    n_points: int = 2000,
    outlet_name: str | None = None,
    title: str | None = None,
    normalized: bool = False,
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
    normalized : bool, default=False
        If True, normalize each component to its maximum value
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
    )
    
    # Set title
    if title is None:
        title = result.experiment_name
    
    # Plot
    return plot_chromatogram_from_df(
        df,
        title=title,
        normalized=normalized,
        **kwargs,
    )


def plot_chromatogram_overlay(
    results: list["SimulationResultWrapper"],
    n_points: int = 2000,
    labels: list[str] | None = None,
    title: str = "Chromatogram Overlay",
    component_filter: list[str] | None = None,
    normalized: bool = False,
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
    component_filter : list[str], optional
        Only plot these components
    normalized : bool, default=False
        If True, normalize each component to its maximum value
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
        )
        chromatograms.append((label, df))
    
    if not chromatograms:
        raise ValueError("No valid results to plot")
    
    return plot_chromatogram_overlay_from_df(
        chromatograms,
        title=title,
        component_filter=component_filter,
        normalized=normalized,
        **kwargs,
    )