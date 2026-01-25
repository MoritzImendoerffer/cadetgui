"""Excel template generator for chromatography experiments.

Generates Excel templates based on selected operation mode, column model,
binding model, and number of components.
"""

from io import BytesIO
from typing import Any
import pandas as pd

from ..operation_modes import (
    BaseOperationMode,
    ParameterDefinition,
    ParameterType,
    SUPPORTED_COLUMN_MODELS,
    SUPPORTED_BINDING_MODELS,
    OPERATION_MODES
)


class ExcelTemplateGenerator:
    """Generates Excel templates for experiment configuration.
    
    The template has two sheets:
    1. Experiments: One row per experiment with process parameters
    2. Column_Binding: Column and binding model parameters (shared across experiments)
    
    Example:
        >>> from cadet_simplified.operation_modes import get_lwe_mode
        >>> mode = get_lwe_mode()
        >>> generator = ExcelTemplateGenerator(
        ...     operation_mode=mode,
        ...     column_model="GeneralRateModel",
        ...     binding_model="StericMassAction",
        ...     n_components=4,
        ...     component_names=["Salt", "Product", "Impurity1", "Impurity2"],
        ... )
        >>> generator.save("template.xlsx")
    """
    
    def __init__(
        self,
        operation_mode: str,
        column_model: str,
        binding_model: str,
        n_components: int,
        component_names: list[str] | None = None,
    ):
        """Initialize template generator.
        
        Parameters
        ----------
        operation_mode : BaseOperationMode
            The operation mode defining process parameters
        column_model : str
            Name of the column model
        binding_model : str
            Name of the binding model
        n_components : int
            Number of components (including salt)
        component_names : list[str], optional
            Names for each component. If not provided, defaults to
            ["Salt", "Component_1", "Component_2", ...]
        """
        
        if column_model not in SUPPORTED_COLUMN_MODELS:
            raise ValueError(f"Unknown column model: {column_model}, supported are {SUPPORTED_COLUMN_MODELS}")
        if binding_model not in SUPPORTED_BINDING_MODELS:
            raise ValueError(f"Unknown binding model: {binding_model}, supported are {SUPPORTED_BINDING_MODELS}")
        if operation_mode not in OPERATION_MODES:
            raise ValueError(f"Unknown operation mode: {operation_mode}, supported are {OPERATION_MODES}")
        if n_components < 2:
            raise ValueError("Need at least 2 components (salt + 1 protein)")
        
        self.operation_mode = OPERATION_MODES[operation_mode]()
        self.column_model = column_model
        self.binding_model = binding_model
        self.n_components = n_components
        
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
            Dictionary with sheet names as keys and DataFrames as values
        """
        return {
            "Experiments": self._generate_experiments_sheet(),
            "Column_Binding": self._generate_column_binding_sheet(),
        }
    
    def save(self, path: str) -> None:
        """Save the template to an Excel file.
        
        Parameters
        ----------
        path : str
            Output file path
        """
        sheets = self.generate()
        
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def to_bytes(self) -> bytes:
        """Generate template as bytes (for web download).
        
        Returns
        -------
        bytes
            Excel file contents
        """
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
        
        # Experiment name (required first column)
        columns.append("experiment_name")
        defaults["experiment_name"] = "experiment_1"
        units["experiment_name"] = "-"
        
        # Add scalar experiment parameters
        for param in self.operation_mode.get_experiment_parameters():
            col_name = param.name
            columns.append(col_name)
            defaults[col_name] = param.default
            units[col_name] = param.unit
        
        # Add per-component experiment parameters
        for param in self.operation_mode.get_component_experiment_parameters():
            for i, comp_name in enumerate(self.component_names):
                # Skip salt for protein-specific parameters like load_concentration
                if i == 0 and param.name in ["load_concentration"]:
                    continue
                
                col_name = f"component_{i+1}_{param.name}"
                columns.append(col_name)
                defaults[col_name] = param.default
                units[col_name] = param.unit
        
        # Add component name columns (for documentation)
        for i, comp_name in enumerate(self.component_names):
            col_name = f"component_{i+1}_name"
            columns.append(col_name)
            defaults[col_name] = comp_name
            units[col_name] = "-"
        
        # Create DataFrame with header row containing units
        # First row is units, second row is example data
        data = []
        
        # Units row (as comment/documentation)
        unit_row = {col: f"[{units.get(col, '-')}]" for col in columns}
        data.append(unit_row)
        
        # Example data row
        example_row = {col: defaults.get(col, "") for col in columns}
        data.append(example_row)
        
        df = pd.DataFrame(data, columns=columns)
        
        return df
    
    def _generate_column_binding_sheet(self) -> pd.DataFrame:
        """Generate the Column_Binding sheet.
        
        Structure: parameter | value | unit | description
        With sections for column scalar, column per-component,
        binding scalar, and binding per-component parameters.
        """
        rows = []
        
        # Header for model info
        rows.append({
            "parameter": "# MODEL SELECTION",
            "value": "",
            "unit": "",
            "description": "",
        })
        rows.append({
            "parameter": "column_model",
            "value": self.column_model,
            "unit": "-",
            "description": "Column model type",
        })
        rows.append({
            "parameter": "binding_model",
            "value": self.binding_model,
            "unit": "-",
            "description": "Binding model type",
        })
        rows.append({
            "parameter": "n_components",
            "value": self.n_components,
            "unit": "-",
            "description": "Number of components",
        })
        
        # Component names
        rows.append({
            "parameter": "# COMPONENT NAMES",
            "value": "",
            "unit": "",
            "description": "",
        })
        for i, name in enumerate(self.component_names):
            rows.append({
                "parameter": f"component_{i+1}_name",
                "value": name,
                "unit": "-",
                "description": f"Name of component {i+1}",
            })
        
        # Column scalar parameters
        rows.append({
            "parameter": "# COLUMN PARAMETERS (SCALAR)",
            "value": "",
            "unit": "",
            "description": "",
        })
        for param in self.operation_mode.get_column_parameters(self.column_model):
            rows.append({
                "parameter": param.name,
                "value": param.default if param.default is not None else "",
                "unit": param.unit,
                "description": param.description,
            })
        
        # Column per-component parameters
        comp_col_params = self.operation_mode.get_component_column_parameters(self.column_model)
        if comp_col_params:
            rows.append({
                "parameter": "# COLUMN PARAMETERS (PER-COMPONENT)",
                "value": "",
                "unit": "",
                "description": "",
            })
            for param in comp_col_params:
                for i, comp_name in enumerate(self.component_names):
                    rows.append({
                        "parameter": f"{param.name}_component_{i+1}",
                        "value": param.default if param.default is not None else "",
                        "unit": param.unit,
                        "description": f"{param.description} for {comp_name}",
                    })
        
        # Binding scalar parameters
        rows.append({
            "parameter": "# BINDING PARAMETERS (SCALAR)",
            "value": "",
            "unit": "",
            "description": "",
        })
        for param in self.operation_mode.get_binding_parameters(self.binding_model):
            rows.append({
                "parameter": param.name,
                "value": param.default if param.default is not None else "",
                "unit": param.unit,
                "description": param.description,
            })
        
        # Binding per-component parameters
        comp_bind_params = self.operation_mode.get_component_binding_parameters(self.binding_model)
        if comp_bind_params:
            rows.append({
                "parameter": "# BINDING PARAMETERS (PER-COMPONENT)",
                "value": "",
                "unit": "",
                "description": "",
            })
            for param in comp_bind_params:
                for i, comp_name in enumerate(self.component_names):
                    rows.append({
                        "parameter": f"{param.name}_component_{i+1}",
                        "value": param.default if param.default is not None else "",
                        "unit": param.unit,
                        "description": f"{param.description} for {comp_name}",
                    })
        
        df = pd.DataFrame(rows)
        return df


def generate_template(
    operation_mode: str | BaseOperationMode,
    column_model: str,
    binding_model: str,
    n_components: int,
    component_names: list[str] | None = None,
    output_path: str | None = None,
) -> bytes | None:
    """Convenience function to generate a template.
    
    Parameters
    ----------
    operation_mode : str or BaseOperationMode
        Operation mode name or instance
    column_model : str
        Column model name
    binding_model : str
        Binding model name
    n_components : int
        Number of components
    component_names : list[str], optional
        Component names
    output_path : str, optional
        If provided, save to this path and return None.
        If not provided, return bytes.
        
    Returns
    -------
    bytes or None
        Excel file contents if output_path is None
    """
    from ..operation_modes import get_operation_mode
    
    if isinstance(operation_mode, str):
        mode = get_operation_mode(operation_mode)
    else:
        mode = operation_mode
    
    generator = ExcelTemplateGenerator(
        operation_mode=mode,
        column_model=column_model,
        binding_model=binding_model,
        n_components=n_components,
        component_names=component_names,
    )
    
    if output_path:
        generator.save(output_path)
        return None
    else:
        return generator.to_bytes()
