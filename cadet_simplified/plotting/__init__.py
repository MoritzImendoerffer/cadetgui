"""Plotting functions for CADET simulation results.

Provides reusable plotting functions that work in Jupyter notebooks
and Panel applications. All functions return HoloViews objects.

Interactive components:
    >>> from cadet_simplified.plotting import InteractiveChromatogram
    >>> widget = InteractiveChromatogram(df, title="Exp 1", conversion_params={...})

Example - Single chromatogram:
    >>> from cadet_simplified.plotting import plot_chromatogram
    >>> plot = plot_chromatogram(result)
    >>> plot  # Display in notebook

Example - Overlay multiple:
    >>> from cadet_simplified.plotting import plot_chromatogram_overlay
    >>> plot = plot_chromatogram_overlay([result1, result2])

Example - From cached DataFrame:
    >>> from cadet_simplified.plotting import plot_chromatogram_from_df
    >>> df = pd.read_parquet("chromatogram.parquet")
    >>> plot = plot_chromatogram_from_df(df)

Example - Inlet profile from process events:
    >>> from cadet_simplified.plotting import plot_inlet_profile
    >>> plot = plot_inlet_profile(process, normalized=True, x_axis="cv")

Example - Convert time to column volumes:
    >>> from cadet_simplified.plotting import time_to_cv, calculate_column_volume_mL
    >>> col_vol = calculate_column_volume_mL(column_parameters)
    >>> cv_array = time_to_cv(time_seconds, flow_rate_mL_min, col_vol)
"""

from .chromatogram import (
    # Unit conversion helpers
    calculate_column_volume_mL,
    time_to_cv,
    get_column_volume_from_process,
    get_flow_rate_from_process,
    # Interpolation
    interpolate_chromatogram,
    get_component_names,
    # Inlet profile (from Process events)
    get_inlet_profile_df,
    plot_inlet_profile,
    # From DataFrame (cached)
    plot_chromatogram_from_df,
    plot_chromatogram_overlay_from_df,
    # From SimulationResultWrapper
    plot_chromatogram,
    plot_chromatogram_overlay,
)

from .interactive import (
    InteractiveChromatogram,
    InteractiveChromatogramOverlay,
)

__all__ = [
    # Interactive components
    "InteractiveChromatogram",
    "InteractiveChromatogramOverlay",
    # Unit conversion helpers
    "calculate_column_volume_mL",
    "time_to_cv",
    "get_column_volume_from_process",
    "get_flow_rate_from_process",
    # Interpolation
    "interpolate_chromatogram",
    "get_component_names",
    # Inlet profile
    "get_inlet_profile_df",
    "plot_inlet_profile",
    # From DataFrame
    "plot_chromatogram_from_df",
    "plot_chromatogram_overlay_from_df",
    # From result
    "plot_chromatogram",
    "plot_chromatogram_overlay",
]