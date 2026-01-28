"""Excel parser for filled experiment templates.

Parses Excel files and converts them to ExperimentConfig and ColumnBindingConfig.

Example:
    >>> result = parse_excel("filled_template.xlsx")
    >>> if result.success:
    ...     for exp in result.experiments:
    ...         process = mode.create_process(exp, result.column_binding)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO
import re

import pandas as pd

from ..core import ExperimentConfig, ColumnBindingConfig, ComponentDefinition


@dataclass
class ParseResult:
    """Result of parsing an Excel template.
    
    Attributes
    ----------
    success : bool
        Whether parsing succeeded
    experiments : list[ExperimentConfig]
        Parsed experiment configurations
    column_binding : ColumnBindingConfig, optional
        Parsed column/binding configuration
    errors : list[str]
        Error messages
    warnings : list[str]
        Warning messages
    """
    success: bool
    experiments: list[ExperimentConfig] = field(default_factory=list)
    column_binding: ColumnBindingConfig | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ExcelParser:
    """Parses filled Excel templates into configuration objects.
    
    Example:
        >>> parser = ExcelParser()
        >>> result = parser.parse("filled_template.xlsx")
        >>> if result.success:
        ...     for exp in result.experiments:
        ...         process = mode.create_process(exp, result.column_binding)
    """
    
    def parse(self, file: str | Path | BinaryIO) -> ParseResult:
        """Parse an Excel template file.
        
        Parameters
        ----------
        file : str, Path, or file-like
            Excel file to parse
            
        Returns
        -------
        ParseResult
            Parsed configuration or errors
        """
        errors = []
        warnings = []
        
        try:
            excel_file = pd.ExcelFile(file)
            sheet_names = excel_file.sheet_names
            
            # Check required sheets
            if "Experiments" not in sheet_names:
                errors.append("Missing required sheet: 'Experiments'")
            if "Column_Binding" not in sheet_names:
                errors.append("Missing required sheet: 'Column_Binding'")
            
            if errors:
                return ParseResult(success=False, errors=errors)
            
            # Parse Column_Binding sheet first
            column_binding, cb_errors, cb_warnings = self._parse_column_binding(
                excel_file.parse("Column_Binding", dtype=str)
            )
            errors.extend(cb_errors)
            warnings.extend(cb_warnings)
            
            if column_binding is None:
                return ParseResult(success=False, errors=errors, warnings=warnings)
            
            # Parse Experiments sheet
            experiments, exp_errors, exp_warnings = self._parse_experiments(
                excel_file.parse("Experiments", dtype=str),
                column_binding,
            )
            errors.extend(exp_errors)
            warnings.extend(exp_warnings)
            
            if errors:
                return ParseResult(
                    success=False,
                    experiments=experiments,
                    column_binding=column_binding,
                    errors=errors,
                    warnings=warnings,
                )
            
            return ParseResult(
                success=True,
                experiments=experiments,
                column_binding=column_binding,
                warnings=warnings,
            )
            
        except Exception as e:
            errors.append(f"Failed to read Excel file: {str(e)}")
            return ParseResult(success=False, errors=errors)
    
    def _parse_column_binding(
        self, df: pd.DataFrame
    ) -> tuple[ColumnBindingConfig | None, list[str], list[str]]:
        """Parse the Column_Binding sheet."""
        errors = []
        warnings = []
        
        # Convert to dict: parameter -> value
        params = {}
        for _, row in df.iterrows():
            param = str(row.get("parameter", "")).strip()
            value = row.get("value")
            
            # Skip header rows and empty rows
            if not param or param.startswith("#"):
                continue
            
            if pd.isna(value) or value is None:
                continue
            
            value = str(value).strip()
            if value == "" or value.lower() == "nan":
                continue
            
            # Convert to appropriate type
            params[param] = self._convert_value(value)
        
        # Extract model selection
        column_model = params.get("column_model")
        binding_model = params.get("binding_model")
        n_components = params.get("n_components")
        
        if not column_model:
            errors.append("Missing column_model in Column_Binding sheet")
        if not binding_model:
            errors.append("Missing binding_model in Column_Binding sheet")
        if not n_components:
            errors.append("Missing n_components in Column_Binding sheet")
        
        if errors:
            return None, errors, warnings
        
        n_components = int(n_components)
        
        # Extract component names
        component_names = []
        for i in range(1, n_components + 1):
            name = params.get(f"component_{i}_name", f"Component_{i}")
            component_names.append(str(name))
        
        # Separate parameters into categories
        column_params = {}
        binding_params = {}
        component_column_params = {}
        component_binding_params = {}
        
        # Pattern for per-component parameters
        component_pattern = re.compile(r"^(.+)_component_(\d+)$")
        
        # Known column scalar parameters (from common JSON configs)
        column_scalar_names = {
            "length", "diameter", "bed_porosity", "particle_porosity",
            "particle_radius", "axial_dispersion", "flow_direction",
        }
        
        # Known binding scalar parameters
        binding_scalar_names = {
            "is_kinetic", "reference_liquid_phase_conc", "reference_solid_phase_conc",
        }
        
        # Known column component parameters
        column_component_names = {
            "film_diffusion", "pore_diffusion", "surface_diffusion", "pore_accessibility",
        }
        
        for param, value in params.items():
            # Skip model selection params
            if param in ("column_model", "binding_model", "n_components"):
                continue
            if param.startswith("component_") and param.endswith("_name"):
                continue
            
            # Check if per-component
            match = component_pattern.match(param)
            if match:
                base_name = match.group(1)
                comp_idx = int(match.group(2)) - 1
                
                # Determine if column or binding parameter
                if base_name in column_component_names or base_name in column_scalar_names:
                    if base_name not in component_column_params:
                        component_column_params[base_name] = [None] * n_components
                    if 0 <= comp_idx < n_components:
                        component_column_params[base_name][comp_idx] = value
                else:
                    if base_name not in component_binding_params:
                        component_binding_params[base_name] = [None] * n_components
                    if 0 <= comp_idx < n_components:
                        component_binding_params[base_name][comp_idx] = value
            else:
                # Scalar parameter
                if param in column_scalar_names:
                    column_params[param] = value
                elif param in binding_scalar_names:
                    binding_params[param] = value
                else:
                    # Guess based on name
                    if any(kw in param.lower() for kw in ["porosity", "length", "diameter", "dispersion", "diffusion", "radius"]):
                        column_params[param] = value
                    else:
                        binding_params[param] = value
        
        # Replace None with 0.0 in component arrays
        for param in component_column_params:
            component_column_params[param] = [
                v if v is not None else 0.0 for v in component_column_params[param]
            ]
        for param in component_binding_params:
            component_binding_params[param] = [
                v if v is not None else 0.0 for v in component_binding_params[param]
            ]
        
        config = ColumnBindingConfig(
            column_model=column_model,
            binding_model=binding_model,
            column_parameters=column_params,
            binding_parameters=binding_params,
            component_column_parameters=component_column_params,
            component_binding_parameters=component_binding_params,
        )
        
        # Store component names for experiments parsing
        config._component_names = component_names
        
        return config, errors, warnings
    
    def _parse_experiments(
        self,
        df: pd.DataFrame,
        column_binding: ColumnBindingConfig,
    ) -> tuple[list[ExperimentConfig], list[str], list[str]]:
        """Parse the Experiments sheet."""
        errors = []
        warnings = []
        experiments = []
        
        component_names = getattr(column_binding, '_component_names', [])
        n_components = len(component_names)
        
        # Skip units row (first row if it contains [unit])
        start_row = 0
        if len(df) > 0:
            first_row = df.iloc[0]
            if any(str(v).startswith("[") for v in first_row.values if pd.notna(v)):
                start_row = 1
        
        # Parse each experiment row
        for idx in range(start_row, len(df)):
            row = df.iloc[idx]
            
            exp_name = row.get("experiment_name")
            if pd.isna(exp_name) or str(exp_name).strip() == "":
                continue
            
            exp_name = str(exp_name).strip()
            
            # Extract parameters
            params = {}
            for col in df.columns:
                if col == "experiment_name":
                    continue
                
                value = row.get(col)
                if pd.isna(value) or value is None:
                    continue
                
                value = str(value).strip()
                if value.startswith("[") or value == "" or value.lower() == "nan":
                    continue
                
                params[col] = self._convert_value(value)
            
            # Build component definitions
            components = []
            for i in range(n_components):
                comp_name = params.get(
                    f"component_{i+1}_name",
                    component_names[i] if i < len(component_names) else f"Component_{i+1}"
                )
                is_salt = (i == 0)
                components.append(ComponentDefinition(name=str(comp_name), is_salt=is_salt))
            
            experiments.append(ExperimentConfig(
                name=exp_name,
                parameters=params,
                components=components,
            ))
        
        if not experiments:
            errors.append("No experiments found in Experiments sheet")
        
        return experiments, errors, warnings
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            numeric = pd.to_numeric(value, errors='coerce')
            if pd.notna(numeric):
                return int(numeric) if numeric == int(numeric) else float(numeric)
            return value


def parse_excel(file: str | Path | BinaryIO) -> ParseResult:
    """Convenience function to parse an Excel file.
    
    Parameters
    ----------
    file : str, Path, or file-like
        Excel file to parse
        
    Returns
    -------
    ParseResult
        Parsed configuration
    """
    parser = ExcelParser()
    return parser.parse(file)
