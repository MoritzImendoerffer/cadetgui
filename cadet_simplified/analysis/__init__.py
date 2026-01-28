"""Analysis module for CADET Simplified.

Provides composable view building for analysis reports with export capability.

Example:
    >>> from cadet_simplified.analysis import AnalysisView
    >>> 
    >>> view = AnalysisView(title="IEX Screening Report", author="Lab Team")
    >>> view.add_section("Chromatogram Analysis")
    >>> view.add_plot(interactive_chromatogram)
    >>> view.add_section("Summary Statistics")
    >>> view.add_table(summary_df)
    >>> 
    >>> # Display in Panel app:
    >>> layout = view.view()
    >>> 
    >>> # Export to HTML:
    >>> html = view.to_html()
    >>> 
    >>> # Or get a download button:
    >>> download_btn = view.download_widget("report.html")
"""

from .view import AnalysisView, ViewComponent

__all__ = [
    "AnalysisView",
    "ViewComponent",
]