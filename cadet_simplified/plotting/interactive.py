"""Interactive chromatogram components with self-contained controls.

Provides Panel-based interactive plot components that bundle controls
(normalization, x-axis selection, dual y-axis) with the chromatogram plot.

Example - Single chromatogram:
    >>> from cadet_simplified.plotting.interactive import InteractiveChromatogram
    >>> widget = InteractiveChromatogram(
    ...     df=chromatogram_df,
    ...     title="Experiment 1",
    ...     conversion_params={"flow_rate_mL_min": 1.0, "column_volume_mL": 0.785},
    ... )
    >>> widget  # Display in notebook or Panel app

Example - Overlay multiple:
    >>> from cadet_simplified.plotting.interactive import InteractiveChromatogramOverlay
    >>> chromatograms = [
    ...     ("Exp 1", df1, {"flow_rate_mL_min": 1.0, "column_volume_mL": 0.785}),
    ...     ("Exp 2", df2, {"flow_rate_mL_min": 0.5, "column_volume_mL": 0.785}),
    ... ]
    >>> widget = InteractiveChromatogramOverlay(chromatograms, title="Comparison")
"""

import colorsys
import panel as pn
import param
import holoviews as hv
import pandas as pd
import numpy as np

from .chromatogram import time_to_cv

hv.extension("bokeh")


# =============================================================================
# Color utilities
# =============================================================================

# Bokeh Category10 palette - well-separated base hues
from bokeh.palettes import Category10, Colorblind

BASE_COLORS = list(Category10[10])
BASE_COLORS = list(Colorblind[8])
def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color to RGB (0-1 range)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return r, g, b


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert RGB (0-1 range) to hex color."""
    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255), int(g * 255), int(b * 255)
    )


def generate_experiment_colors(
    n_experiments: int,
    n_components: int,
    saturation_end: float = 0.25,
    value_end: float = 1.0,
) -> list[list[str]]:
    """Generate colors for experiments with varying saturation/value per component.
    
    Uses the approach from StackOverflow (ImportanceOfBeingErnest):
    - Each experiment gets a distinct base hue from Category10
    - Components vary in saturation (original → 0.25) and value (original → 1.0)
    - First component = original saturated color, last = lighter/desaturated
    
    Parameters
    ----------
    n_experiments : int
        Number of experiments
    n_components : int
        Number of components per experiment
    saturation_end : float
        Target saturation for last component (0-1)
    value_end : float
        Target value/brightness for last component (0-1)
        
    Returns
    -------
    list[list[str]]
        Colors[experiment_idx][component_idx] = hex color
    """
    colors = []
    
    for exp_idx in range(n_experiments):
        base_hex = BASE_COLORS[exp_idx % len(BASE_COLORS)]
        r, g, b = hex_to_rgb(base_hex)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        exp_colors = []
        for comp_idx in range(n_components):
            if n_components == 1:
                # Single component: use original color
                new_s, new_v = s, v
            else:
                # Interpolate saturation and value
                t = comp_idx / (n_components - 1)
                new_s = s + t * (saturation_end - s)
                new_v = v + t * (value_end - v)
            
            new_r, new_g, new_b = colorsys.hsv_to_rgb(h, new_s, new_v)
            exp_colors.append(rgb_to_hex(new_r, new_g, new_b))
        
        colors.append(exp_colors)
    
    return colors


class InteractiveChromatogram(pn.viewable.Viewer):
    """Single chromatogram with normalize/x-axis/dual-axis controls.
    
    Controls appear above the plot in three horizontal cards and reactively 
    update the visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Chromatogram data with 'time' column (in seconds) and component columns
    title : str
        Plot title
    conversion_params : dict, optional
        Dictionary with "flow_rate_mL_min" and "column_volume_mL" keys.
        If None, CV option is disabled.
    width : int, default=800
        Plot width in pixels
    height : int, default=400
        Plot height in pixels
    """
    
    normalized = param.Boolean(default=False, doc="Normalize each component to max")
    x_axis = param.Selector(
        default="minutes",
        objects={"Time (min)": "minutes", "Column Volumes (CV)": "cv"},
        doc="X-axis unit",
    )
    dual_axis = param.Boolean(default=False, doc="Use separate y-axes")
    left_axis_components = param.List(default=[], doc="Components on left y-axis")
    
    def __init__(
        self,
        df: pd.DataFrame,
        title: str = "Chromatogram",
        conversion_params: dict | None = None,
        width: int = 800,
        height: int = 400,
        **params,
    ):
        super().__init__(**params)
        
        self._df = df.copy()
        self._title = title
        self._conversion_params = conversion_params
        self._width = width
        self._height = height
        
        # Get component names
        self._component_names = self._get_component_names()
        
        # Set default left axis to first component
        if self._component_names and not self.left_axis_components:
            self.left_axis_components = [self._component_names[0]]
        
        # Validate conversion params
        self._cv_available = self._validate_conversion_params(conversion_params)
        
        # Build controls
        self._build_controls()
    
    def _get_component_names(self) -> list[str]:
        """Extract component names from DataFrame."""
        return [c for c in self._df.columns if "time" not in c.lower()]
    
    def _validate_conversion_params(self, params: dict | None) -> bool:
        """Check if conversion params are valid for CV calculation."""
        if params is None:
            return False
        flow_rate = params.get("flow_rate_mL_min")
        col_vol = params.get("column_volume_mL")
        return flow_rate is not None and col_vol is not None and col_vol > 0
    
    def _build_controls(self):
        """Build control widgets in three horizontal cards."""
        # Display card
        self._normalize_checkbox = pn.widgets.Checkbox.from_param(
            self.param.normalized,
            name="Normalize",
        )
        
        # X-Axis card
        self._x_axis_select = pn.widgets.Select.from_param(
            self.param.x_axis,
            name="",
            width=150,
        )
        
        # Disable CV option if conversion params not available
        if not self._cv_available:
            self._x_axis_select.disabled = True
            self._x_axis_select.value = "minutes"
        
        # Y-Axis card
        self._dual_axis_checkbox = pn.widgets.Checkbox.from_param(
            self.param.dual_axis,
            name="Dual Axis",
        )
        
        self._left_axis_select = pn.widgets.MultiChoice.from_param(
            self.param.left_axis_components,
            options=self._component_names,
            name="Left Axis",
            width=200,
            solid=False,
        )
    
    @param.depends("dual_axis", watch=True)
    def _update_left_axis_visibility(self):
        """Show/hide left axis selector based on dual axis checkbox."""
        # This is handled in __panel__ via pn.bind
        pass
    
    def _transform_data(self) -> pd.DataFrame:
        """Transform data based on current control settings."""
        df = self._df.copy()
        
        # Find time column
        time_cols = [c for c in df.columns if "time" in c.lower()]
        if not time_cols:
            raise ValueError("No time column found in DataFrame")
        time_col = time_cols[0]
        
        # Convert x-axis
        if self.x_axis == "cv" and self._cv_available:
            df[time_col] = time_to_cv(
                df[time_col].values,
                self._conversion_params["flow_rate_mL_min"],
                self._conversion_params["column_volume_mL"],
            )
        else:
            # Convert seconds to minutes
            df[time_col] = df[time_col] / 60.0
        
        # Get component columns
        component_cols = [c for c in df.columns if "time" not in c.lower()]
        
        # Normalize if requested (each component to its own max)
        if self.normalized:
            for comp in component_cols:
                max_val = df[comp].max()
                if max_val > 0:
                    df[comp] = df[comp] / max_val
        
        return df
    
    @param.depends("normalized", "x_axis", "dual_axis", "left_axis_components")
    def _create_plot(self) -> hv.Overlay:
        """Create the plot with current settings."""
        import hvplot.pandas  # noqa: F401
        
        df = self._transform_data()
        
        # Find time column
        time_cols = [c for c in df.columns if "time" in c.lower()]
        time_col = time_cols[0]
        
        # Get component columns
        component_cols = [c for c in df.columns if "time" not in c.lower()]
        
        # Determine labels
        if self.x_axis == "cv" and self._cv_available:
            xlabel = "Column Volumes (CV)"
        else:
            xlabel = "Time (min)"
        
        ylabel = "Normalized Concentration (-)" if self.normalized else "Concentration (mM)"
        
        # Determine which components go on which axis
        left_components = set(self.left_axis_components) if self.dual_axis else set()
        
        # Create plots for each component
        plots = []
        for comp in component_cols:
            is_left = comp in left_components
            
            # Line style: dashed for left axis, solid for right
            line_dash = "dashed" if is_left and self.dual_axis else "solid"
            
            p = df.hvplot.line(
                x=time_col,
                y=comp,
                label=comp,
            )
            
            # Apply options
            opts_kwargs = {
                "line_dash": line_dash,
            }
            
            # Assign to right axis if dual axis enabled and not a left component
            if self.dual_axis and not is_left:
                opts_kwargs["yaxis"] = "right"
            
            p = p.opts(**opts_kwargs)
            plots.append(p)
        
        # Overlay
        if len(plots) == 1:
            overlay = plots[0]
        else:
            overlay = plots[0]
            for p in plots[1:]:
                overlay = overlay * p
        
        # Base options
        opts_kwargs = {
            "xlabel": xlabel,
            "ylabel": ylabel,
            "title": self._title,
            "width": self._width,
            "height": self._height,
            "legend_position": "right",
            "show_legend": True,
            "framewise": True,
            "tools": ["hover", "pan", "wheel_zoom", "box_zoom", "reset", "save"],
            "active_tools": ["wheel_zoom"],
        }
        
        # Enable multi_y if dual axis
        if self.dual_axis:
            opts_kwargs["multi_y"] = True
        
        overlay = overlay.opts(**opts_kwargs)
        
        return overlay
    
    def get_current_plot(self) -> hv.Overlay:
        """Get plot with current control settings (for export)."""
        return self._create_plot()
    
    def get_current_state(self) -> dict:
        """Get current settings for serialization."""
        return {
            "normalized": self.normalized,
            "x_axis": self.x_axis,
            "dual_axis": self.dual_axis,
            "left_axis_components": self.left_axis_components,
            "title": self._title,
            "cv_available": self._cv_available,
        }
    
    def __panel__(self):
        """Return the Panel layout."""
        # Display card
        display_card = pn.Card(
            self._normalize_checkbox,
            title="Display",
            sizing_mode="fixed",
            width=120,
        )
        
        # X-Axis card
        x_axis_card = pn.Card(
            self._x_axis_select,
            title="X-Axis",
            sizing_mode="fixed",
            width=180,
        )
        
        # Y-Axis card content (dynamic based on dual_axis)
        def y_axis_content():
            if self.dual_axis:
                return pn.Column(
                    self._dual_axis_checkbox,
                    self._left_axis_select,
                    sizing_mode="stretch_width",
                )
            else:
                return pn.Column(
                    self._dual_axis_checkbox,
                    sizing_mode="stretch_width",
                )
        
        y_axis_card = pn.Card(
            pn.bind(lambda _: y_axis_content(), self.param.dual_axis),
            title="Y-Axis",
            sizing_mode="stretch_width",
        )
        
        controls = pn.Row(
            display_card,
            x_axis_card,
            y_axis_card,
            sizing_mode="stretch_width",
        )
        
        plot_pane = pn.pane.HoloViews(
            self._create_plot,
            sizing_mode="stretch_width",
        )
        
        return pn.Column(
            controls,
            plot_pane,
            sizing_mode="stretch_width",
        )


class InteractiveChromatogramOverlay(pn.viewable.Viewer):
    """Multiple chromatograms with shared controls.
    
    Controls appear above the plot in three horizontal cards. When CV mode is 
    selected, each experiment is converted using its own conversion parameters. 
    If any experiment lacks parameters, a warning is shown and the plot falls 
    back to minutes.
    
    Parameters
    ----------
    chromatograms : list[tuple[str, pd.DataFrame, dict | None]]
        List of (label, dataframe, conversion_params) tuples.
        - label: Display name for the experiment
        - dataframe: Chromatogram data with 'time' column (seconds)
        - conversion_params: Dict with "flow_rate_mL_min" and "column_volume_mL",
          or None if not available
    title : str
        Plot title
    component_filter : list[str], optional
        Only show these components. If None, show all.
    width : int, default=900
        Plot width in pixels
    height : int, default=450
        Plot height in pixels
    """
    
    normalized = param.Boolean(default=False, doc="Normalize each component to max")
    x_axis = param.Selector(
        default="minutes",
        objects={"Time (min)": "minutes", "Column Volumes (CV)": "cv"},
        doc="X-axis unit",
    )
    dual_axis = param.Boolean(default=False, doc="Use separate y-axes")
    left_axis_components = param.List(default=[], doc="Components on left y-axis")
    
    def __init__(
        self,
        chromatograms: list[tuple[str, pd.DataFrame, dict | None]],
        title: str = "Chromatogram Overlay",
        component_filter: list[str] | None = None,
        width: int = 900,
        height: int = 450,
        **params,
    ):
        super().__init__(**params)
        
        self._chromatograms = chromatograms
        self._title = title
        self._component_filter = component_filter
        self._width = width
        self._height = height
        
        # Get union of all component names
        self._component_names = self._get_all_component_names()
        
        # Set default left axis to first component
        if self._component_names and not self.left_axis_components:
            self.left_axis_components = [self._component_names[0]]
        
        # Check which experiments have valid conversion params
        self._cv_status = self._check_cv_availability()
        
        # Build controls
        self._build_controls()
    
    def _get_all_component_names(self) -> list[str]:
        """Get union of all component names across chromatograms."""
        all_names = set()
        for label, df, params in self._chromatograms:
            component_cols = [c for c in df.columns if "time" not in c.lower()]
            all_names.update(component_cols)
        
        # Apply filter if specified
        if self._component_filter is not None:
            all_names = all_names & set(self._component_filter)
        
        # Return sorted list for consistent ordering
        return sorted(list(all_names))
    
    def _check_cv_availability(self) -> dict:
        """Check CV availability for each experiment."""
        status = {
            "all_available": True,
            "any_available": False,
            "missing": [],
        }
        
        for label, df, params in self._chromatograms:
            if params is None:
                status["all_available"] = False
                status["missing"].append(label)
            else:
                flow_rate = params.get("flow_rate_mL_min")
                col_vol = params.get("column_volume_mL")
                if flow_rate is None or col_vol is None or col_vol <= 0:
                    status["all_available"] = False
                    status["missing"].append(label)
                else:
                    status["any_available"] = True
        
        return status
    
    def _build_controls(self):
        """Build control widgets in three horizontal cards."""
        # Display card
        self._normalize_checkbox = pn.widgets.Checkbox.from_param(
            self.param.normalized,
            name="Normalize",
        )
        
        # X-Axis card
        self._x_axis_select = pn.widgets.Select.from_param(
            self.param.x_axis,
            name="",
            width=150,
        )
        
        # Disable CV option if no experiments have conversion params
        if not self._cv_status["any_available"]:
            self._x_axis_select.disabled = True
            self._x_axis_select.value = "minutes"
        
        # Y-Axis card
        self._dual_axis_checkbox = pn.widgets.Checkbox.from_param(
            self.param.dual_axis,
            name="Dual Axis",
        )
        
        self._left_axis_select = pn.widgets.MultiChoice.from_param(
            self.param.left_axis_components,
            options=self._component_names,
            name="Left Axis",
            width=200,
            solid=False,
        )
        
        # Warning pane (shown when CV selected but some missing)
        self._warning_pane = pn.pane.Alert(
            "",
            alert_type="warning",
            visible=False,
            sizing_mode="stretch_width",
        )
    
    def _get_effective_x_axis(self) -> str:
        """Get effective x-axis, falling back to minutes if CV not fully available."""
        if self.x_axis == "cv" and not self._cv_status["all_available"]:
            return "minutes"
        return self.x_axis
    
    def _update_warning(self):
        """Update warning visibility and message."""
        if self.x_axis == "cv" and not self._cv_status["all_available"]:
            missing = ", ".join(self._cv_status["missing"])
            self._warning_pane.object = (
                f"Cannot use Column Volumes: missing conversion parameters for {missing}. "
                "Showing time in minutes."
            )
            self._warning_pane.visible = True
        else:
            self._warning_pane.visible = False
    
    @param.depends("normalized", "x_axis", "dual_axis", "left_axis_components")
    def _create_plot(self) -> hv.Overlay:
        """Create the overlay plot with current settings."""
        import hvplot.pandas  # noqa: F401
        
        # Update warning
        self._update_warning()
        
        effective_x_axis = self._get_effective_x_axis()
        
        # Determine labels
        if effective_x_axis == "cv":
            xlabel = "Column Volumes (CV)"
        else:
            xlabel = "Time (min)"
        
        ylabel = "Normalized Concentration (-)" if self.normalized else "Concentration (mM)"
        
        # Determine which components go on which axis
        left_components = set(self.left_axis_components) if self.dual_axis else set()
        
        # Generate color palette: experiment-based hue, component-based lightness
        n_experiments = len(self._chromatograms)
        n_components = len(self._component_names)
        color_palette = generate_experiment_colors(n_experiments, n_components)
        
        plots = []
        
        for exp_idx, (label, df, conv_params) in enumerate(self._chromatograms):
            _df = df.copy()
            
            # Find time column
            time_cols = [c for c in _df.columns if "time" in c.lower()]
            if not time_cols:
                continue
            time_col = time_cols[0]
            
            # Convert x-axis
            if effective_x_axis == "cv" and conv_params is not None:
                flow_rate = conv_params.get("flow_rate_mL_min")
                col_vol = conv_params.get("column_volume_mL")
                if flow_rate and col_vol and col_vol > 0:
                    _df[time_col] = time_to_cv(_df[time_col].values, flow_rate, col_vol)
                else:
                    _df[time_col] = _df[time_col] / 60.0
            else:
                _df[time_col] = _df[time_col] / 60.0
            
            # Get component columns
            component_cols = [c for c in _df.columns if "time" not in c.lower()]
            
            # Apply filter if specified
            if self._component_filter is not None:
                component_cols = [c for c in component_cols if c in self._component_filter]
            
            # Normalize if requested (each component to its own max)
            if self.normalized:
                for comp in component_cols:
                    max_val = _df[comp].max()
                    if max_val > 0:
                        _df[comp] = _df[comp] / max_val
            
            # Create plot for each component
            for comp in component_cols:
                full_label = f"{label} - {comp}"
                is_left = comp in left_components
                
                # Line style: dashed for left axis, solid for right
                line_dash = "dashed" if is_left and self.dual_axis else "solid"
                
                # Get color based on experiment and component index
                comp_idx = self._component_names.index(comp) if comp in self._component_names else 0
                color = color_palette[exp_idx][comp_idx]
                
                p = _df.hvplot.line(
                    x=time_col,
                    y=comp,
                    label=full_label,
                )
                
                # Apply options
                opts_kwargs = {
                    "line_dash": line_dash,
                    "color": color,
                }
                
                # Assign to right axis if dual axis enabled and not a left component
                if self.dual_axis and not is_left:
                    opts_kwargs["yaxis"] = "right"
                
                p = p.opts(**opts_kwargs)
                plots.append(p)
        
        if not plots:
            # Return empty plot with message
            return hv.Text(0.5, 0.5, "No data to display").opts(
                title=self._title,
                width=self._width,
                height=self._height,
            )
        
        # Overlay all plots
        overlay = plots[0]
        for p in plots[1:]:
            overlay = overlay * p
        
        # Base options
        opts_kwargs = {
            "xlabel": xlabel,
            "ylabel": ylabel,
            "title": self._title,
            "width": self._width,
            "height": self._height,
            "legend_position": "right",
            "show_legend": True,
            "framewise": True,
            "tools": ["hover", "pan", "wheel_zoom", "box_zoom", "reset", "save"],
            "active_tools": ["wheel_zoom"],
        }
        
        # Enable multi_y if dual axis
        if self.dual_axis:
            opts_kwargs["multi_y"] = True
        
        overlay = overlay.opts(**opts_kwargs)
        
        return overlay
    
    def get_current_plot(self) -> hv.Overlay:
        """Get plot with current control settings (for export)."""
        return self._create_plot()
    
    def get_current_state(self) -> dict:
        """Get current settings for serialization."""
        return {
            "normalized": self.normalized,
            "x_axis": self.x_axis,
            "effective_x_axis": self._get_effective_x_axis(),
            "dual_axis": self.dual_axis,
            "left_axis_components": self.left_axis_components,
            "title": self._title,
            "n_experiments": len(self._chromatograms),
            "cv_status": self._cv_status,
        }
    
    def __panel__(self):
        """Return the Panel layout."""
        # Display card
        display_card = pn.Card(
            self._normalize_checkbox,
            title="Display",
            sizing_mode="fixed",
            width=120,
        )
        
        # X-Axis card
        x_axis_card = pn.Card(
            self._x_axis_select,
            title="X-Axis",
            sizing_mode="fixed",
            width=180,
        )
        
        # Y-Axis card content (dynamic based on dual_axis)
        def y_axis_content():
            if self.dual_axis:
                return pn.Column(
                    self._dual_axis_checkbox,
                    self._left_axis_select,
                    sizing_mode="stretch_width",
                )
            else:
                return pn.Column(
                    self._dual_axis_checkbox,
                    sizing_mode="stretch_width",
                )
        
        y_axis_card = pn.Card(
            pn.bind(lambda _: y_axis_content(), self.param.dual_axis),
            title="Y-Axis",
            sizing_mode="stretch_width",
        )
        
        controls = pn.Row(
            display_card,
            x_axis_card,
            y_axis_card,
            sizing_mode="stretch_width",
        )
        
        plot_pane = pn.pane.HoloViews(
            self._create_plot,
            sizing_mode="stretch_width",
        )
        
        return pn.Column(
            controls,
            self._warning_pane,
            plot_pane,
            sizing_mode="stretch_width",
        )