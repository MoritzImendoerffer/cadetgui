"""Plotting functions for CADET simulation results.

Provides reusable plotting functions that work in Jupyter notebooks
and Panel applications. All functions return HoloViews objects.

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
    >>> plot = plot_inlet_profile(process, normalized=True)
"""

from .chromatogram import (
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

__all__ = [
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