"""Excel template generation and parsing.

Example - Generate template:
    >>> from cadet_simplified.excel import ExcelTemplateGenerator
    >>> generator = ExcelTemplateGenerator(
    ...     operation_mode="LWE_concentration_based",
    ...     column_model="LumpedRateModelWithPores",
    ...     binding_model="StericMassAction",
    ...     n_components=3,
    ... )
    >>> generator.save("template.xlsx")

Example - Parse filled template:
    >>> from cadet_simplified.excel import parse_excel
    >>> result = parse_excel("filled_template.xlsx")
    >>> if result.success:
    ...     for exp in result.experiments:
    ...         print(exp.name)
"""

from .template_generator import ExcelTemplateGenerator
from .parser import ExcelParser, ParseResult, parse_excel

__all__ = [
    'ExcelTemplateGenerator',
    'ExcelParser',
    'ParseResult',
    'parse_excel',
]
