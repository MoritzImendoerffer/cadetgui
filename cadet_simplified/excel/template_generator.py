"""Excel template generator for chromatography experiments.

Generates Excel templates based on:
- Operation mode (defines process parameters)
- Column model (parameters from JSON config)
- Binding model (parameters from JSON config)
- Number of components

Example:
    >>> generator = ExcelTemplateGenerator(
    ...     operation_mode="LWE_concentration_based",
    ...     column_model="LumpedRateModelWithPores",
    ...     binding_model="StericMassAction",
    ...     n_components=3,
    ...     component_names=["Salt", "Product", "Impurity"],
    ... )
    >>> generator.save("template.xlsx")
"""

from io import BytesIO
from typing import Any

import pandas as pd

from ..configs import (
    get_binding_model_config,
    get_column_model_config,
    list_binding_models,
    list_column_models,
)
from ..operation_modes import get_operation_mode, list_operation_modes


class ExcelTemplateGenerator:
    """Generates Excel templates for experiment configuration.
    
    The template has two sheets:
    1. Experiments: One row per experiment with process parameters
    2. Column_Binding: Column and binding model parameters (shared)
    
    Parameters
    ----------
    operation_mode : str
        Operation mode name (e.g., "LWE_concentration_based")
    column_model : str
        Column model name (e.g., "LumpedRateModelWithPores")
    binding_model : str
        Binding model name (e.g., "StericMassAction")
    n_components : int
        Number of components (including salt)
    component_names : list[str], optional
        Names for components. Defaults to ["Salt", "Component_1", ...]
    """
    
    def __init__(
        self,
        operation_mode: str,
        column_model: str,
        binding_model: str,
        n_components: int,
        component_names: list[str] | None = None,
    ):
        # Validate inputs
        if operation_mode not in list_operation_modes():
            raise ValueError(f"Unknown operation mode: {operation_mode}")
        if column_model not in list_column_models():
            raise ValueError(f"Unknown column model: {column_model}")
        if binding_model not in list_binding_models():
            raise ValueError(f"Unknown binding model: {binding_model}")
        if n_components < 2:
            raise ValueError("Need at least 2 components (salt + 1 protein)")
        
        self.operation_mode = get_operation_mode(operation_mode)
        self.column_model = column_model
        self.binding_model = binding_model
        self.n_components = n_components
        
        # Get configs from JSON
        self.column_config = get_column_model_config(column_model)
        self.binding_config = get_binding_model_config(binding_model)
        
        # Generate default component names if not provided
        if component_names is None:
            component_names = ["Salt"] + [f"Component_{i}" for i in range(1, n_components)]
        elif len(component_names) != n_components:
            raise ValueError(f"Expected {n_components} component names, got {len(component_names)}")
        
        self.component_names = component_names
    
    def generate(self) -> dict[str, pd.DataFrame]:
        """Generate the template as DataFrames.
        
        Returns
        -------
        dict[str, pd.DataFrame]
            Sheet names -> DataFrames
        """
        return {
            "Experiments": self._generate_experiments_sheet(),
            "Column_Binding": self._generate_column_binding_sheet(),
        }
    
    def save(self, path: str) -> None:
        """Save the template to an Excel file."""
        sheets = self.generate()
        
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def to_bytes(self) -> bytes:
        """Generate template as bytes (for web download)."""
        sheets = self.generate()
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def _generate_experiments_sheet(self) -> pd.DataFrame:
        """Generate the Experiments sheet."""
        columns = []
        defaults = {}
        units = {}
        
        # Experiment name (first column)
        columns.append("experiment_name")
        defaults["experiment_name"] = "experiment_1"
        units["experiment_name"] = "-"
        
        # Scalar experiment parameters from operation mode
        for param in self.operation_mode.get_experiment_parameters():
            columns.append(param.name)
            defaults[param.name] = param.default
            units[param.name] = param.unit
        
        # Per-component experiment parameters
        for param in self.operation_mode.get_component_experiment_parameters():
            for i, comp_name in enumerate(self.component_names):
                # Skip salt for protein-specific parameters
                if i == 0 and param.name in ["load_concentration"]:
                    continue
                
                col_name = f"component_{i+1}_{param.name}"
                columns.append(col_name)
                defaults[col_name] = param.default
                units[col_name] = param.unit
        
        # Component name columns
        for i, comp_name in enumerate(self.component_names):
            col_name = f"component_{i+1}_name"
            columns.append(col_name)
            defaults[col_name] = comp_name
            units[col_name] = "-"
        
        # Create DataFrame: first row = units, second row = example
        data = []
        
        # Units row
        unit_row = {col: f"[{units.get(col, '-')}]" for col in columns}
        data.append(unit_row)
        
        # Example row
        example_row = {col: defaults.get(col, "") for col in columns}
        data.append(example_row)
        
        return pd.DataFrame(data, columns=columns)
    
    def _generate_column_binding_sheet(self) -> pd.DataFrame:
        """Generate the Column_Binding sheet."""
        rows = []
        
        # Model selection header
        rows.append({"parameter": "# MODEL SELECTION", "value": "", "unit": "", "description": ""})
        rows.append({"parameter": "column_model", "value": self.column_model, "unit": "-", "description": "Column model type"})
        rows.append({"parameter": "binding_model", "value": self.binding_model, "unit": "-", "description": "Binding model type"})
        rows.append({"parameter": "n_components", "value": self.n_components, "unit": "-", "description": "Number of components"})
        
        # Component names
        rows.append({"parameter": "# COMPONENT NAMES", "value": "", "unit": "", "description": ""})
        for i, name in enumerate(self.component_names):
            rows.append({
                "parameter": f"component_{i+1}_name",
                "value": name,
                "unit": "-",
                "description": f"Name of component {i+1}",
            })
        
        # Column scalar parameters
        rows.append({"parameter": "# COLUMN PARAMETERS (SCALAR)", "value": "", "unit": "", "description": ""})
        for param in self.column_config.scalar_parameters:
            rows.append({
                "parameter": param.name,
                "value": param.default if param.default is not None else "",
                "unit": param.unit,
                "description": param.description,
            })
        
        # Column per-component parameters
        if self.column_config.component_parameters:
            rows.append({"parameter": "# COLUMN PARAMETERS (PER-COMPONENT)", "value": "", "unit": "", "description": ""})
            for param in self.column_config.component_parameters:
                for i, comp_name in enumerate(self.component_names):
                    rows.append({
                        "parameter": f"{param.name}_component_{i+1}",
                        "value": param.default if param.default is not None else "",
                        "unit": param.unit,
                        "description": f"{param.description} for {comp_name}",
                    })
        
        # Binding scalar parameters
        rows.append({"parameter": "# BINDING PARAMETERS (SCALAR)", "value": "", "unit": "", "description": ""})
        for param in self.binding_config.scalar_parameters:
            rows.append({
                "parameter": param.name,
                "value": param.default if param.default is not None else "",
                "unit": param.unit,
                "description": param.description,
            })
        
        # Binding per-component parameters
        if self.binding_config.component_parameters:
            rows.append({"parameter": "# BINDING PARAMETERS (PER-COMPONENT)", "value": "", "unit": "", "description": ""})
            for param in self.binding_config.component_parameters:
                for i, comp_name in enumerate(self.component_names):
                    rows.append({
                        "parameter": f"{param.name}_component_{i+1}",
                        "value": param.default if param.default is not None else "",
                        "unit": param.unit,
                        "description": f"{param.description} for {comp_name}",
                    })
        
        return pd.DataFrame(rows)
