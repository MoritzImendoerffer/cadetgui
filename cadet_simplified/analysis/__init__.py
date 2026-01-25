"""Analysis module for simulation results.

Provides:
- AnalysisView: Composable view builder for displaying analysis results
- BaseAnalysis: Abstract base class for analysis types
- SimpleChromatogramAnalysis: Overlay plot + summary table
- DetailedAnalysis: Per-experiment plots + parameter comparison

Example:
    >>> from cadet_simplified.analysis import AnalysisView, get_analysis
    >>> from cadet_simplified.storage import FileResultsStorage
    >>> 
    >>> # Load experiments
    >>> storage = FileResultsStorage("./experiments")
    >>> loaded = storage.load_results_by_selection([...], n_workers=4)
    >>> 
    >>> # Run analysis
    >>> view = AnalysisView()
    >>> analysis = get_analysis("simple")
    >>> analysis.run(loaded, view)
    >>> 
    >>> # Display (in Panel app)
    >>> layout = view.view()
"""

from .view import AnalysisView
from .analyses import (
    BaseAnalysis,
    SimpleChromatogramAnalysis,
    DetailedAnalysis,
    ANALYSIS_REGISTRY,
    get_analysis,
    list_analyses,
)

__all__ = [
    # View builder
    'AnalysisView',
    # Analysis base and implementations
    'BaseAnalysis',
    'SimpleChromatogramAnalysis',
    'DetailedAnalysis',
    # Registry functions
    'ANALYSIS_REGISTRY',
    'get_analysis',
    'list_analyses',
]
