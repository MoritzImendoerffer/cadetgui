"""Reference data configuration for parameter fitting.

Reference data contains experimental chromatograms that are compared
against simulations during the fitting process.

Example
-------
    import numpy as np
    from cadet_simplified.optimization import ReferenceDataConfig
    
    # Create reference data
    ref = ReferenceDataConfig(
        experiment_name="experiment_1",
        time=np.array([0, 60, 120, 180, 240]),  # seconds
        concentrations={
            "Salt": np.array([50.0, 50.0, 52.0, 55.0, 60.0]),
            "Product": np.array([0.0, 0.0, 0.1, 0.5, 0.3]),
        }
    )
    
    # Convert to DataFrame
    df = ref.to_dataframe()
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from CADETProcess.reference import ReferenceIO


@dataclass
class ReferenceDataConfig:
    """Reference chromatogram data for fitting.
    
    Attributes
    ----------
    experiment_name : str
        Name of the experiment this reference belongs to
    time : np.ndarray
        Time points in seconds
    concentrations : dict[str, np.ndarray]
        Component name -> concentration array mapping
        
    Notes
    -----
    Time should be in seconds to match CADET internal units.
    Concentrations should be in the same units as the simulation (typically mM).
    """
    
    experiment_name: str
    time: np.ndarray
    concentrations: dict[str, np.ndarray]
    
    def __post_init__(self):
        """Validate the reference data."""
        n_time = len(self.time)
        for comp_name, conc in self.concentrations.items():
            if len(conc) != n_time:
                raise ValueError(
                    f"Component '{comp_name}' has {len(conc)} values "
                    f"but time has {n_time} points"
                )
    
    @property
    def n_points(self) -> int:
        """Number of time points."""
        return len(self.time)
    
    @property
    def component_names(self) -> list[str]:
        """List of component names."""
        return list(self.concentrations.keys())
    
    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return float(self.time[-1] - self.time[0])
    
    @property
    def duration_minutes(self) -> float:
        """Duration in minutes."""
        return self.duration_seconds / 60.0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with 'time' column and one column per component
        """
        data = {"time": self.time}
        data.update(self.concentrations)
        return pd.DataFrame(data)
    
    def to_reference_io(self, component: str) -> "ReferenceIO":
        """Convert to CADET-Process ReferenceIO for one component.
        
        Parameters
        ----------
        component : str
            Component name
            
        Returns
        -------
        ReferenceIO
            CADET-Process reference object
        """
        from CADETProcess.reference import ReferenceIO
        
        if component not in self.concentrations:
            raise ValueError(
                f"Component '{component}' not in reference data. "
                f"Available: {self.component_names}"
            )
        
        return ReferenceIO(
            name=f"{self.experiment_name}_{component}",
            time=self.time.copy(),
            solution=self.concentrations[component].copy(),
        )
    
    def to_reference_io_multi(self, components: list[str] | None = None) -> "ReferenceIO":
        """Convert to CADET-Process ReferenceIO for multiple components.
        
        Parameters
        ----------
        components : list[str], optional
            Component names to include. If None, includes all components.
            
        Returns
        -------
        ReferenceIO
            CADET-Process reference object with multi-component solution
        """
        from CADETProcess.reference import ReferenceIO
        
        if components is None:
            components = self.component_names
        
        # Stack concentrations: shape (n_time, n_components)
        solution = np.column_stack([
            self.concentrations[name] for name in components
        ])
        
        return ReferenceIO(
            name=self.experiment_name,
            time=self.time.copy(),
            solution=solution,
        )
    
    def get_concentration(self, component: str, time_minutes: float | None = None) -> np.ndarray | float:
        """Get concentration data for a component.
        
        Parameters
        ----------
        component : str
            Component name
        time_minutes : float, optional
            If provided, interpolate to get concentration at this time
            
        Returns
        -------
        np.ndarray or float
            Concentration array or interpolated value
        """
        if component not in self.concentrations:
            raise ValueError(f"Component '{component}' not found")
        
        conc = self.concentrations[component]
        
        if time_minutes is None:
            return conc
        
        # Interpolate
        time_seconds = time_minutes * 60.0
        return np.interp(time_seconds, self.time, conc)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, experiment_name: str) -> "ReferenceDataConfig":
        """Create from a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'time' column and component columns
        experiment_name : str
            Name for this experiment
            
        Returns
        -------
        ReferenceDataConfig
            Reference data configuration
        """
        if "time" not in df.columns:
            raise ValueError("DataFrame must have a 'time' column")
        
        time = df["time"].values.astype(float)
        concentrations = {}
        
        for col in df.columns:
            if col != "time":
                concentrations[col] = df[col].values.astype(float)
        
        return cls(
            experiment_name=experiment_name,
            time=time,
            concentrations=concentrations,
        )
    
    def __repr__(self) -> str:
        return (
            f"ReferenceDataConfig({self.experiment_name}, "
            f"{self.n_points} points, "
            f"components={self.component_names})"
        )


def parse_reference_sheet(
    df: pd.DataFrame,
    experiment_name: str,
    component_names: list[str],
    time_column: str = "time",
) -> tuple[ReferenceDataConfig | None, list[str], list[str]]:
    """Parse a reference data sheet from Excel.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from Excel sheet
    experiment_name : str
        Name of the experiment
    component_names : list[str]
        Expected component names from the process
    time_column : str
        Name of the time column
        
    Returns
    -------
    ReferenceDataConfig or None
        Parsed reference data, or None if parsing failed
    list[str]
        Error messages
    list[str]
        Warning messages
    """
    errors = []
    warnings = []
    
    # Check for time column (case-insensitive)
    time_col_found = None
    for col in df.columns:
        if col.lower() == time_column.lower():
            time_col_found = col
            break
    
    if time_col_found is None:
        errors.append(f"Missing '{time_column}' column")
        return None, errors, warnings
    
    # Extract time
    try:
        time = df[time_col_found].values.astype(float)
    except (ValueError, TypeError) as e:
        errors.append(f"Could not parse time column: {e}")
        return None, errors, warnings
    
    # Match component columns (case-insensitive)
    concentrations = {}
    component_map = {name.lower(): name for name in component_names}
    
    for col in df.columns:
        if col == time_col_found:
            continue
        
        col_lower = col.lower()
        
        if col_lower in component_map:
            # Exact match (case-insensitive)
            matched_name = component_map[col_lower]
            try:
                concentrations[matched_name] = df[col].values.astype(float)
            except (ValueError, TypeError) as e:
                errors.append(f"Could not parse column '{col}': {e}")
        else:
            # Column doesn't match any component
            warnings.append(
                f"Column '{col}' does not match any component {component_names}, skipping"
            )
    
    if not concentrations:
        errors.append(f"No valid component columns found. Expected: {component_names}")
        return None, errors, warnings
    
    # Check for missing components (just a warning, not an error)
    found_components = set(concentrations.keys())
    expected_components = set(component_names)
    missing = expected_components - found_components
    if missing:
        warnings.append(f"Missing reference data for components: {list(missing)}")
    
    try:
        ref_data = ReferenceDataConfig(
            experiment_name=experiment_name,
            time=time,
            concentrations=concentrations,
        )
        return ref_data, errors, warnings
    except ValueError as e:
        errors.append(str(e))
        return None, errors, warnings
