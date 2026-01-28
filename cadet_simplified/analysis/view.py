"""Composable view builder for analysis results with export capability.

AnalysisView provides a fluent interface for building analysis displays
that can contain text, plots, tables, and dividers. Supports export to
standalone HTML with interactive Bokeh plots.

Example:
    >>> view = AnalysisView(title="IEX Screening Report", author="Lab Team")
    >>> view.add_section("Chromatogram Analysis")
    >>> view.add_plot(interactive_chromatogram)
    >>> view.add_section("Summary Statistics")
    >>> view.add_table(summary_df)
    >>> 
    >>> # In Panel app:
    >>> layout = view.view()
    >>> 
    >>> # Export to HTML:
    >>> html = view.to_html()
    >>> 
    >>> # Or get a download button:
    >>> download_btn = view.download_widget("report.html")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
import io
import re

import panel as pn
import pandas as pd
import holoviews as hv

from bokeh.embed import components as bokeh_components
from bokeh.resources import CDN as bokeh_cdn


ComponentType = Literal[
    "title", "section", "markdown", "plot", "table", 
    "divider", "alert", "spacer", "raw_html"
]


@dataclass
class ViewComponent:
    """Structured storage for a view component.
    
    Attributes
    ----------
    type : str
        Component type identifier
    data : Any
        Raw data for the component:
        - markdown/title/section/alert: str
        - table: pd.DataFrame
        - plot: plot object (hv.Overlay, InteractiveChromatogram, etc.)
        - divider/spacer: None
    options : dict
        Rendering options (level, alert_type, height, etc.)
    """
    type: ComponentType
    data: Any
    options: dict = field(default_factory=dict)


class AnalysisView:
    """Composable view builder for analysis displays with export capability.
    
    Builds a Panel Column by appending components (text, plots, tables, dividers).
    Supports export to standalone HTML with interactive Bokeh plots.
    
    Parameters
    ----------
    title : str, optional
        Report title (appears in export header)
    author : str, optional
        Report author
    description : str, optional
        Report description
        
    Example
    -------
    >>> view = AnalysisView(title="My Report")
    >>> view.add_section("Results")
    >>> view.add_text("The experiment showed...")
    >>> view.add_plot(chromatogram_plot)
    >>> view.add_table(results_df)
    >>> 
    >>> # Display in Panel
    >>> view.view()
    >>> 
    >>> # Export
    >>> html = view.to_html()
    """
    
    def __init__(
        self,
        title: str | None = None,
        author: str | None = None,
        description: str | None = None,
    ):
        self.title = title
        self.author = author
        self.description = description
        self.created_at = datetime.now()
        self._components: list[ViewComponent] = []
    
    def clear(self) -> "AnalysisView":
        """Clear all components from the view.
        
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components = []
        return self
    
    # -------------------------------------------------------------------------
    # Content Addition Methods
    # -------------------------------------------------------------------------
    
    def add_title(self, text: str, level: int = 1) -> "AnalysisView":
        """Add a heading (h1-h6).
        
        Parameters
        ----------
        text : str
            Heading text
        level : int, default=1
            Heading level (1-6)
            
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        level = max(1, min(6, level))  # Clamp to 1-6
        self._components.append(ViewComponent(
            type="title",
            data=text,
            options={"level": level},
        ))
        return self
    
    def add_section(self, title: str) -> "AnalysisView":
        """Add a section header with divider above.
        
        Convenience method that adds a divider followed by an h2 heading.
        
        Parameters
        ----------
        title : str
            Section title
            
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components.append(ViewComponent(
            type="section",
            data=title,
            options={},
        ))
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
        self._components.append(ViewComponent(
            type="markdown",
            data=markdown,
            options={},
        ))
        return self
    
    def add_plot(
        self,
        plot: Any,
        height: int = 400,
    ) -> "AnalysisView":
        """Add a plot (hvplot, bokeh, InteractiveChromatogram, etc.).
        
        Parameters
        ----------
        plot : Any
            Plot object. Supported types:
            - HoloViews/hvplot objects
            - Bokeh figures
            - InteractiveChromatogram / InteractiveChromatogramOverlay
        height : int, default=400
            Plot height in pixels
            
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components.append(ViewComponent(
            type="plot",
            data=plot,
            options={"height": height},
        ))
        return self
    
    def add_table(
        self,
        df: pd.DataFrame,
        height: int = 200,
        show_index: bool = False,
    ) -> "AnalysisView":
        """Add a data table.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to display
        height : int, default=200
            Table height in pixels
        show_index : bool, default=False
            Whether to show DataFrame index
            
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components.append(ViewComponent(
            type="table",
            data=df.copy(),
            options={"height": height, "show_index": show_index},
        ))
        return self
    
    def add_divider(self) -> "AnalysisView":
        """Add a horizontal divider.
        
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components.append(ViewComponent(
            type="divider",
            data=None,
            options={},
        ))
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
        self._components.append(ViewComponent(
            type="spacer",
            data=None,
            options={"height": height},
        ))
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
        self._components.append(ViewComponent(
            type="alert",
            data=message,
            options={"alert_type": alert_type},
        ))
        return self
    
    def add_raw_html(self, html: str) -> "AnalysisView":
        """Add raw HTML content (for export only, rendered as markdown in Panel).
        
        Parameters
        ----------
        html : str
            Raw HTML string
            
        Returns
        -------
        AnalysisView
            Self for method chaining
        """
        self._components.append(ViewComponent(
            type="raw_html",
            data=html,
            options={},
        ))
        return self
    
    # -------------------------------------------------------------------------
    # Display Methods
    # -------------------------------------------------------------------------
    
    @property
    def is_empty(self) -> bool:
        """Check if view has no components."""
        return len(self._components) == 0
    
    def __len__(self) -> int:
        """Return number of components."""
        return len(self._components)
    
    def _component_to_panel(self, comp: ViewComponent) -> Any:
        """Convert a ViewComponent to a Panel object for display."""
        if comp.type == "title":
            level = comp.options.get("level", 1)
            prefix = "#" * level
            return pn.pane.Markdown(
                f"{prefix} {comp.data}",
                sizing_mode="stretch_width",
            )
        
        elif comp.type == "section":
            return pn.Column(
                pn.layout.Divider(),
                pn.pane.Markdown(f"## {comp.data}", sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
            )
        
        elif comp.type == "markdown":
            return pn.pane.Markdown(comp.data, sizing_mode="stretch_width")
        
        elif comp.type == "plot":
            height = comp.options.get("height", 400)
            plot = comp.data
            
            # Check if it's an interactive component with __panel__ method
            if hasattr(plot, "__panel__"):
                return plot
            
            # HoloViews/hvplot
            plot_type = type(plot).__module__.split(".")[0]
            if plot_type in ("holoviews", "hvplot"):
                return pn.pane.HoloViews(plot, height=height, sizing_mode="stretch_width")
            elif plot_type == "bokeh":
                return pn.pane.Bokeh(plot, height=height, sizing_mode="stretch_width")
            elif plot_type == "matplotlib":
                return pn.pane.Matplotlib(plot, height=height, sizing_mode="stretch_width", tight=True)
            else:
                return pn.panel(plot, height=height, sizing_mode="stretch_width")
        
        elif comp.type == "table":
            height = comp.options.get("height", 200)
            show_index = comp.options.get("show_index", False)
            return pn.widgets.Tabulator(
                comp.data,
                height=height,
                show_index=show_index,
                sizing_mode="stretch_width",
                disabled=True,
            )
        
        elif comp.type == "divider":
            return pn.layout.Divider()
        
        elif comp.type == "spacer":
            height = comp.options.get("height", 20)
            return pn.Spacer(height=height)
        
        elif comp.type == "alert":
            alert_type = comp.options.get("alert_type", "info")
            return pn.pane.Alert(comp.data, alert_type=alert_type, sizing_mode="stretch_width")
        
        elif comp.type == "raw_html":
            return pn.pane.HTML(comp.data, sizing_mode="stretch_width")
        
        else:
            return pn.pane.Markdown(f"*Unknown component type: {comp.type}*")
    
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
                sizing_mode="stretch_width",
            )
        
        panel_objects = [self._component_to_panel(comp) for comp in self._components]
        
        return pn.Column(
            *panel_objects,
            sizing_mode="stretch_width",
        )
    
    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------
    
    def _markdown_to_html(self, text: str) -> str:
        """Convert simple markdown to HTML.
        
        Handles: headers, bold, italic, code, links, lists.
        """
        html = text
        
        # Headers (must be at start of line)
        html = re.sub(r"^###### (.+)$", r"<h6>\1</h6>", html, flags=re.MULTILINE)
        html = re.sub(r"^##### (.+)$", r"<h5>\1</h5>", html, flags=re.MULTILINE)
        html = re.sub(r"^#### (.+)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
        html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
        
        # Bold and italic
        html = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", html)
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
        
        # Inline code
        html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
        
        # Links
        html = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', html)
        
        # Unordered lists (simple)
        lines = html.split("\n")
        in_list = False
        result_lines = []
        for line in lines:
            if line.strip().startswith("- "):
                if not in_list:
                    result_lines.append("<ul>")
                    in_list = True
                result_lines.append(f"<li>{line.strip()[2:]}</li>")
            else:
                if in_list:
                    result_lines.append("</ul>")
                    in_list = False
                result_lines.append(line)
        if in_list:
            result_lines.append("</ul>")
        html = "\n".join(result_lines)
        
        # Paragraphs (wrap non-tagged content)
        lines = html.split("\n")
        result_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                result_lines.append("")
            elif stripped.startswith("<"):
                result_lines.append(line)
            else:
                result_lines.append(f"<p>{line}</p>")
        html = "\n".join(result_lines)
        
        return html
    
    def _plot_to_html(self, plot: Any) -> str:
        """Convert a plot to embeddable HTML."""
        # Handle interactive chromatogram components
        if hasattr(plot, "get_current_plot"):
            plot = plot.get_current_plot()
        
        # Convert HoloViews to Bokeh
        if hasattr(plot, "opts"):  # HoloViews object
            renderer = hv.renderer("bokeh")
            plot = renderer.get_plot(plot).state
        
        # Get Bokeh HTML components
        script, div = bokeh_components(plot)
        
        return f"{div}\n{script}"
    
    def _table_to_html(self, df: pd.DataFrame, show_index: bool = False) -> str:
        """Convert DataFrame to HTML table."""
        return df.to_html(
            index=show_index,
            classes="table table-striped table-hover",
            border=0,
        )
    
    def _alert_type_to_bootstrap(self, alert_type: str) -> str:
        """Map alert type to Bootstrap class."""
        mapping = {
            "info": "alert-info",
            "success": "alert-success",
            "warning": "alert-warning",
            "danger": "alert-danger",
        }
        return mapping.get(alert_type, "alert-info")
    
    def _component_to_html(self, comp: ViewComponent) -> str:
        """Convert a ViewComponent to HTML for export."""
        if comp.type == "title":
            level = comp.options.get("level", 1)
            return f"<h{level}>{comp.data}</h{level}>"
        
        elif comp.type == "section":
            return f'<hr class="my-4">\n<h2>{comp.data}</h2>'
        
        elif comp.type == "markdown":
            return self._markdown_to_html(comp.data)
        
        elif comp.type == "plot":
            try:
                return f'<div class="plot-container my-3">{self._plot_to_html(comp.data)}</div>'
            except Exception as e:
                return f'<div class="alert alert-warning">Could not render plot: {e}</div>'
        
        elif comp.type == "table":
            show_index = comp.options.get("show_index", False)
            return f'<div class="table-responsive my-3">{self._table_to_html(comp.data, show_index)}</div>'
        
        elif comp.type == "divider":
            return '<hr class="my-4">'
        
        elif comp.type == "spacer":
            height = comp.options.get("height", 20)
            return f'<div style="height: {height}px;"></div>'
        
        elif comp.type == "alert":
            alert_type = comp.options.get("alert_type", "info")
            bs_class = self._alert_type_to_bootstrap(alert_type)
            return f'<div class="alert {bs_class}" role="alert">{comp.data}</div>'
        
        elif comp.type == "raw_html":
            return comp.data
        
        else:
            return f"<!-- Unknown component type: {comp.type} -->"
    
    def to_html(self, embed_resources: bool = False) -> str:
        """Export view as standalone HTML.
        
        Parameters
        ----------
        embed_resources : bool, default=False
            If True, embed Bokeh JS/CSS inline (larger file, works offline).
            If False, use CDN links (smaller file, requires internet).
            
        Returns
        -------
        str
            Complete HTML document
        """
        # Build header
        header_parts = []
        if self.title:
            header_parts.append(f"<h1>{self.title}</h1>")
        
        meta_parts = []
        if self.author:
            meta_parts.append(f"Author: {self.author}")
        meta_parts.append(f"Generated: {self.created_at.strftime('%Y-%m-%d %H:%M')}")
        
        if meta_parts:
            header_parts.append(f'<p class="text-muted">{" | ".join(meta_parts)}</p>')
        
        if self.description:
            header_parts.append(f"<p>{self.description}</p>")
        
        header_html = "\n".join(header_parts)
        if header_html:
            header_html = f'<header class="mb-4">{header_html}</header>'
        
        # Build body
        body_parts = [self._component_to_html(comp) for comp in self._components]
        body_html = "\n".join(body_parts)
        
        # Bokeh resources
        if embed_resources:
            bokeh_css = bokeh_cdn.render_css()
            bokeh_js = bokeh_cdn.render_js()
        else:
            bokeh_css = "\n".join(f'<link rel="stylesheet" href="{url}">' for url in bokeh_cdn.css_files)
            bokeh_js = "\n".join(f'<script src="{url}"></script>' for url in bokeh_cdn.js_files)
        
        # Complete HTML document
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title or "Analysis Report"}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Bokeh Resources -->
    {bokeh_css}
    {bokeh_js}
    
    <style>
        body {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        .plot-container {{
            margin: 1rem 0;
            width: 100%;
        }}
        
        /* Make Bokeh plots responsive */
        .plot-container > div,
        .plot-container .bk-root,
        .plot-container .bk {{
            width: 100% !important;
            max-width: 100% !important;
        }}
        .table-responsive {{
            margin: 1rem 0;
        }}
        code {{
            background-color: #f8f9fa;
            padding: 0.2em 0.4em;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        {header_html}
        <main>
            {body_html}
        </main>
        <footer class="mt-5 pt-3 border-top text-muted">
            <small>Generated by CADET Simplified</small>
        </footer>
    </div>
</body>
</html>"""
        
        return html
    
    def download_widget(
        self,
        filename: str = "report.html",
        button_type: str = "primary",
        label: str = "Download Report",
    ) -> pn.widgets.FileDownload:
        """Return a download button that exports current state.
        
        Parameters
        ----------
        filename : str, default="report.html"
            Download filename
        button_type : str, default="primary"
            Button style (primary, success, warning, danger, light, dark)
        label : str, default="Download Report"
            Button label
            
        Returns
        -------
        pn.widgets.FileDownload
            Download button widget
        """
        def get_html():
            html_content = self.to_html()
            return io.BytesIO(html_content.encode("utf-8"))
        
        return pn.widgets.FileDownload(
            callback=get_html,
            filename=filename,
            button_type=button_type,
            label=label,
        )