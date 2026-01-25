"""Composable view builder for analysis results.

AnalysisView provides a fluent interface for building analysis displays
that can contain text, plots, tables, and dividers. Each add_* method
appends a new row to the view.

Example:
    >>> view = AnalysisView()
    >>> view.add_text("## Chromatogram Analysis")
    >>> view.add_plot(chromatogram_hvplot)
    >>> view.add_divider()
    >>> view.add_text("### Summary Statistics")
    >>> view.add_table(summary_df)
    >>> 
    >>> # In Panel app:
    >>> layout = view.view()
"""

from typing import Any

import panel as pn
import pandas as pd


class AnalysisView:
    """Composable view builder for analysis displays.
    
    Builds a Panel Column by appending components (text, plots, tables, dividers).
    Each component is added as a new row in the final layout.
    
    Attributes
    ----------
    _components : list
        List of Panel components to be rendered
    """
    
    def __init__(self):
        """Initialize empty view."""
        self._components: list[Any] = []
    
    def clear(self) -> "AnalysisView":
        """Clear all components from the view.
        
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components = []
        return self
    
    def add_text(self, markdown: str) -> "AnalysisView":
        """Add a markdown text block.
        
        Parameters
        ----------
        markdown : str
            Markdown-formatted text
            
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components.append(
            pn.pane.Markdown(markdown, sizing_mode='stretch_width')
        )
        return self
    
    def add_plot(
        self,
        plot: Any,
        height: int = 400,
        sizing_mode: str = 'stretch_width',
    ) -> "AnalysisView":
        """Add a plot (hvplot, bokeh, matplotlib, etc.).
        
        Parameters
        ----------
        plot : Any
            Plot object (hvplot, bokeh figure, matplotlib figure, etc.)
        height : int, default=400
            Plot height in pixels
        sizing_mode : str, default='stretch_width'
            Panel sizing mode
            
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        # Detect plot type and wrap appropriately
        plot_type = type(plot).__module__.split('.')[0]
        
        if plot_type == 'holoviews' or plot_type == 'hvplot':
            # hvplot/holoviews - use HoloViews pane
            self._components.append(
                pn.pane.HoloViews(plot, height=height, sizing_mode=sizing_mode)
            )
        elif plot_type == 'bokeh':
            # Bokeh figure
            self._components.append(
                pn.pane.Bokeh(plot, height=height, sizing_mode=sizing_mode)
            )
        elif plot_type == 'matplotlib':
            # Matplotlib figure
            self._components.append(
                pn.pane.Matplotlib(plot, height=height, sizing_mode=sizing_mode, tight=True)
            )
        else:
            # Generic - let Panel figure it out
            self._components.append(
                pn.panel(plot, height=height, sizing_mode=sizing_mode)
            )
        
        return self
    
    def add_table(
        self,
        df: pd.DataFrame,
        height: int = 200,
        show_index: bool = False,
        **tabulator_kwargs,
    ) -> "AnalysisView":
        """Add a data table (Tabulator widget).
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to display
        height : int, default=200
            Table height in pixels
        show_index : bool, default=False
            Whether to show DataFrame index
        **tabulator_kwargs
            Additional arguments passed to Tabulator
            
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components.append(
            pn.widgets.Tabulator(
                df,
                height=height,
                show_index=show_index,
                sizing_mode='stretch_width',
                disabled=True,  # Read-only
                **tabulator_kwargs,
            )
        )
        return self
    
    def add_divider(self) -> "AnalysisView":
        """Add a horizontal divider.
        
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components.append(pn.layout.Divider())
        return self
    
    def add_spacer(self, height: int = 20) -> "AnalysisView":
        """Add vertical spacing.
        
        Parameters
        ----------
        height : int, default=20
            Spacer height in pixels
            
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components.append(pn.Spacer(height=height))
        return self
    
    def add_row(self, *components, **kwargs) -> "AnalysisView":
        """Add multiple components in a horizontal row.
        
        Parameters
        ----------
        *components
            Components to place side by side
        **kwargs
            Arguments passed to pn.Row
            
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components.append(
            pn.Row(*components, sizing_mode='stretch_width', **kwargs)
        )
        return self
    
    def add_alert(
        self,
        message: str,
        alert_type: str = "info",
    ) -> "AnalysisView":
        """Add an alert/notification box.
        
        Parameters
        ----------
        message : str
            Alert message
        alert_type : str, default="info"
            One of: "info", "success", "warning", "danger"
            
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components.append(
            pn.pane.Alert(message, alert_type=alert_type, sizing_mode='stretch_width')
        )
        return self
    
    def add_component(self, component: Any) -> "AnalysisView":
        """Add an arbitrary Panel component.
        
        Parameters
        ----------
        component : Any
            Any Panel-compatible component
            
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components.append(component)
        return self
    
    @property
    def is_empty(self) -> bool:
        """Check if view has no components."""
        return len(self._components) == 0
    
    def view(self) -> pn.Column:
        """Build and return the Panel Column layout.
        
        Returns
        -------
        pn.Column
            Panel Column containing all added components
        """
        if not self._components:
            return pn.Column(
                pn.pane.Markdown("*No analysis results to display*"),
                sizing_mode='stretch_width',
            )
        
        return pn.Column(
            *self._components,
            sizing_mode='stretch_width',
        )
    
    def __len__(self) -> int:
        """Return number of components."""
        return len(self._components)
