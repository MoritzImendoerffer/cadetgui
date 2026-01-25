"""Analysis types for simulation results.

Defines the BaseAnalysis interface and concrete implementations.
Each analysis type processes loaded experiments and populates an AnalysisView.

Example:
    >>> from cadet_simplified.analysis import SimpleChromatogramAnalysis, AnalysisView
    >>> from cadet_simplified.storage import FileResultsStorage
    >>> 
    >>> storage = FileResultsStorage("./experiments")
    >>> loaded = storage.load_results_by_selection([...])
    >>> 
    >>> view = AnalysisView()
    >>> analysis = SimpleChromatogramAnalysis()
    >>> analysis.run(loaded, view)
    >>> 
    >>> # Display in Panel
    >>> view.view()
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

from .view import AnalysisView

if TYPE_CHECKING:
    from ..storage.interfaces import LoadedExperiment


class BaseAnalysis(ABC):
    """Abstract base class for analysis types.
    
    Subclasses implement specific analysis workflows that process
    loaded experiments and populate an AnalysisView.
    
    Attributes
    ----------
    name : str
        Short identifier for the analysis type
    description : str
        Human-readable description shown in UI
    """
    
    name: str = "base"
    description: str = "Base analysis"
    
    @abstractmethod
    def run(
        self,
        experiments: list["LoadedExperiment"],
        view: AnalysisView,
    ) -> None:
        """Execute the analysis and populate the view.
        
        Parameters
        ----------
        experiments : list[LoadedExperiment]
            Loaded experiments with results
        view : AnalysisView
            View to populate with results
        """
        pass
    
    def validate(self, experiments: list["LoadedExperiment"]) -> list[str]:
        """Validate experiments before analysis.
        
        Parameters
        ----------
        experiments : list[LoadedExperiment]
            Experiments to validate
            
        Returns
        -------
        list[str]
            List of validation errors (empty if valid)
        """
        errors = []
        if not experiments:
            errors.append("No experiments selected for analysis")
        return errors


class SimpleChromatogramAnalysis(BaseAnalysis):
    """Simple chromatogram overlay analysis.
    
    Creates:
    - Overlay plot of all chromatograms (hvplot + bokeh)
    - Summary table with experiment metadata
    """
    
    name = "simple"
    description = "Chromatogram overlay with summary table"
    
    def run(
        self,
        experiments: list["LoadedExperiment"],
        view: AnalysisView,
    ) -> None:
        """Create chromatogram overlay and summary table."""
        errors = self.validate(experiments)
        if errors:
            for error in errors:
                view.add_alert(error, alert_type="danger")
            return
        
        # Header
        view.add_text("## Chromatogram Overlay")
        view.add_text(f"*{len(experiments)} experiment(s) selected*")
        
        # Create overlay plot
        try:
            plot = self._create_overlay_plot(experiments)
            view.add_plot(plot, height=450)
        except ImportError as e:
            view.add_alert(
                f"hvplot not available for interactive plotting: {e}. "
                "Install with: pip install hvplot",
                alert_type="warning"
            )
            # Fallback to matplotlib if hvplot not available
            try:
                plot = self._create_matplotlib_plot(experiments)
                view.add_plot(plot, height=450)
            except Exception as e2:
                view.add_alert(f"Could not create plot: {e2}", alert_type="danger")
        except Exception as e:
            view.add_alert(f"Error creating plot: {e}", alert_type="danger")
        
        view.add_divider()
        
        # Summary table
        view.add_text("### Selected Experiments")
        summary_df = self._create_summary_table(experiments)
        view.add_table(summary_df, height=min(200, 50 + len(experiments) * 30))
    
    def _create_overlay_plot(self, experiments: list["LoadedExperiment"]):
        """Create hvplot overlay of chromatograms."""
        import hvplot.pandas  # noqa: F401
        
        plots = []
        
        for exp in experiments:
            if exp.chromatogram_df is None:
                continue
            
            df = exp.chromatogram_df.copy()
            
            # Get component columns (everything except 'time')
            component_cols = [c for c in df.columns if c != 'time']
            
            # Convert time to minutes for readability
            df['time_min'] = df['time'] / 60.0
            
            # Create label for legend
            label_prefix = f"{exp.experiment_set_name}/{exp.experiment_name}"
            
            # Plot each component
            for comp in component_cols:
                label = f"{label_prefix} - {comp}"
                plot = df.hvplot.line(
                    x='time_min',
                    y=comp,
                    label=label,
                )
                plots.append(plot)
        
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
            width=800,
            height=400,
            tools=['hover', 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'],
            active_tools=['wheel_zoom'],
        )
        
        return overlay
    
    def _create_matplotlib_plot(self, experiments: list["LoadedExperiment"]):
        """Fallback matplotlib plot if hvplot not available."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for exp in experiments:
            if exp.chromatogram_df is None:
                continue
            
            df = exp.chromatogram_df
            time_min = df['time'] / 60.0
            
            component_cols = [c for c in df.columns if c != 'time']
            label_prefix = f"{exp.experiment_set_name}/{exp.experiment_name}"
            
            for comp in component_cols:
                ax.plot(time_min, df[comp], label=f"{label_prefix} - {comp}")
        
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Concentration (mM)')
        ax.set_title('Chromatogram Overlay')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_summary_table(self, experiments: list["LoadedExperiment"]) -> pd.DataFrame:
        """Create summary table of selected experiments."""
        rows = []
        
        for exp in experiments:
            row = {
                "Experiment Set": exp.experiment_set_name,
                "Experiment": exp.experiment_name,
                "Column Model": exp.column_binding.column_model,
                "Binding Model": exp.column_binding.binding_model,
            }
            
            # Add some key parameters if available
            params = exp.experiment_config.parameters
            if "flow_rate_mL_min" in params:
                row["Flow Rate (mL/min)"] = params["flow_rate_mL_min"]
            if "gradient_start_mM" in params:
                row["Gradient Start (mM)"] = params["gradient_start_mM"]
            if "gradient_end_mM" in params:
                row["Gradient End (mM)"] = params["gradient_end_mM"]
            
            # Add result info
            if exp.result.success:
                row["Status"] = "✓ Success"
                row["Runtime (s)"] = f"{exp.result.runtime_seconds:.2f}"
            else:
                row["Status"] = "✗ Failed"
                row["Runtime (s)"] = "-"
            
            rows.append(row)
        
        return pd.DataFrame(rows)


class DetailedAnalysis(BaseAnalysis):
    """Detailed per-experiment analysis.
    
    Creates:
    - Individual chromatogram for each experiment
    - Parameter comparison table
    - Basic statistics
    """
    
    name = "detailed"
    description = "Individual chromatograms with parameter comparison"
    
    def run(
        self,
        experiments: list["LoadedExperiment"],
        view: AnalysisView,
    ) -> None:
        """Create detailed analysis for each experiment."""
        errors = self.validate(experiments)
        if errors:
            for error in errors:
                view.add_alert(error, alert_type="danger")
            return
        
        view.add_text("## Detailed Analysis")
        view.add_text(f"*{len(experiments)} experiment(s) selected*")
        view.add_divider()
        
        # Parameter comparison table
        view.add_text("### Parameter Comparison")
        params_df = self._create_params_comparison(experiments)
        view.add_table(params_df, height=min(300, 50 + len(experiments) * 30))
        
        view.add_divider()
        
        # Individual chromatograms
        view.add_text("### Individual Chromatograms")
        
        for exp in experiments:
            view.add_text(f"#### {exp.experiment_set_name} / {exp.experiment_name}")
            
            if exp.chromatogram_df is not None:
                try:
                    plot = self._create_single_plot(exp)
                    view.add_plot(plot, height=300)
                except Exception as e:
                    view.add_alert(f"Could not create plot: {e}", alert_type="warning")
            else:
                view.add_alert("No chromatogram data available", alert_type="info")
            
            view.add_spacer(10)
    
    def _create_params_comparison(self, experiments: list["LoadedExperiment"]) -> pd.DataFrame:
        """Create table comparing parameters across experiments."""
        # Collect all unique parameter keys
        all_keys = set()
        for exp in experiments:
            all_keys.update(exp.experiment_config.parameters.keys())
        
        # Build comparison rows
        rows = []
        for exp in experiments:
            row = {
                "Experiment Set": exp.experiment_set_name,
                "Experiment": exp.experiment_name,
            }
            for key in sorted(all_keys):
                value = exp.experiment_config.parameters.get(key)
                if isinstance(value, float):
                    row[key] = f"{value:.4g}"
                else:
                    row[key] = str(value) if value is not None else "-"
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_single_plot(self, exp: "LoadedExperiment"):
        """Create plot for a single experiment."""
        import hvplot.pandas  # noqa: F401
        
        df = exp.chromatogram_df.copy()
        df['time_min'] = df['time'] / 60.0
        
        component_cols = [c for c in df.columns if c not in ('time', 'time_min')]
        
        # Melt to long format for easier plotting
        df_long = df.melt(
            id_vars=['time_min'],
            value_vars=component_cols,
            var_name='Component',
            value_name='Concentration',
        )
        
        plot = df_long.hvplot.line(
            x='time_min',
            y='Concentration',
            by='Component',
            xlabel='Time (min)',
            ylabel='Concentration (mM)',
            title=f'{exp.experiment_name}',
            legend='right',
            height=280,
            width=700,
        )
        
        return plot


# Analysis registry
ANALYSIS_REGISTRY: dict[str, type[BaseAnalysis]] = {
    "simple": SimpleChromatogramAnalysis,
    "detailed": DetailedAnalysis,
}


def get_analysis(name: str) -> BaseAnalysis:
    """Get an analysis instance by name.
    
    Parameters
    ----------
    name : str
        Analysis name (key in ANALYSIS_REGISTRY)
        
    Returns
    -------
    BaseAnalysis
        Analysis instance
        
    Raises
    ------
    ValueError
        If name not found in registry
    """
    if name not in ANALYSIS_REGISTRY:
        available = list(ANALYSIS_REGISTRY.keys())
        raise ValueError(f"Unknown analysis: {name}. Available: {available}")
    
    return ANALYSIS_REGISTRY[name]()


def list_analyses() -> list[dict[str, str]]:
    """List all available analyses.
    
    Returns
    -------
    list[dict]
        List of {"name": ..., "description": ...} dicts
    """
    return [
        {"name": cls.name, "description": cls.description}
        for cls in ANALYSIS_REGISTRY.values()
    ]
