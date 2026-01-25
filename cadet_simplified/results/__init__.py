"""Results export utilities.

For persistent storage of results, use FileResultsStorage from the storage module.
This module provides export utilities for creating shareable files.

Example - Export from storage:
    >>> from cadet_simplified.storage import FileResultsStorage
    >>> from cadet_simplified.results import ResultsExporter
    >>> 
    >>> storage = FileResultsStorage("./experiments")
    >>> loaded = storage.load_results_by_selection([...])
    >>> 
    >>> exporter = ResultsExporter()
    >>> exporter.export_to_excel(loaded, "./exports/analysis.xlsx")
"""

from .analyzer import (
    ResultsExporter,
    InterpolatedChromatogram,
    # Backwards compatibility
    ResultsAnalyzer,
)

__all__ = [
    'ResultsExporter',
    'InterpolatedChromatogram',
    # Backwards compatibility
    'ResultsAnalyzer',
]
