"""Load-Wash-Elute (LWE) operation mode with concentration-based gradient.

This operation mode uses a single inlet with concentration switching/gradients,
similar to how real chromatography systems work.

Example:
    >>> from cadet_simplified.operation_modes import get_operation_mode
    >>> from cadet_simplified.core import ExperimentConfig, ColumnBindingConfig
    >>> 
    >>> mode = get_operation_mode("LWE_concentration_based")
    >>> process = mode.create_process(experiment_config, column_binding_config)
"""

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from CADETProcess.processModel import Process

from .base import BaseOperationMode, ProcessParameterDef
from ..core import ExperimentConfig, ColumnBindingConfig


class LWEConcentrationBased(BaseOperationMode):
    """Load-Wash-Elute process with concentration-based gradient elution.
    
    This mode uses a single inlet and modulates concentrations over time:
    - Equilibration: Column equilibrated at wash salt concentration
    - Load: Sample loaded at load salt concentration
    - Wash: Unbound material washed out at wash salt concentration
    - Elute: Linear salt gradient from wash to elution concentration
    - Strip (optional): High salt strip to clean column
    
    All volumes are specified in Column Volumes (CV) for lab convenience.
    Flow rate is specified in mL/min.
    Salt concentrations in mM.
    """
    
    name = "LWE_concentration_based"
    description = "Load-Wash-Elute with linear salt gradient elution"
    
    def get_experiment_parameters(self) -> list[ProcessParameterDef]:
        """Get experiment parameters in lab-friendly units."""
        return [
            # Flow
            ProcessParameterDef(
                name="flow_rate_mL_min",
                display_name="Flow Rate",
                unit="mL/min",
                description="Volumetric flow rate",
                default=1.0,
                bounds=(0.01, 10.0),
            ),
            # Volumes (all in CV)
            ProcessParameterDef(
                name="equilibration_cv",
                display_name="Equilibration Volume",
                unit="CV",
                description="Volume for initial column equilibration",
                default=5.0,
                required=False,
                bounds=(0.0, 50.0),
            ),
            ProcessParameterDef(
                name="load_cv",
                display_name="Load Volume",
                unit="CV",
                description="Sample loading volume",
                default=5.0,
                bounds=(0.1, 100.0),
            ),
            ProcessParameterDef(
                name="wash_cv",
                display_name="Wash Volume",
                unit="CV",
                description="Post-load wash volume",
                default=5.0,
                bounds=(0.0, 50.0),
            ),
            ProcessParameterDef(
                name="elution_cv",
                display_name="Elution Volume",
                unit="CV",
                description="Gradient elution volume",
                default=20.0,
                bounds=(1.0, 100.0),
            ),
            ProcessParameterDef(
                name="strip_cv",
                display_name="Strip Volume",
                unit="CV",
                description="High salt strip volume (0 to skip)",
                default=0.0,
                required=False,
                bounds=(0.0, 20.0),
            ),
            # Salt concentrations
            ProcessParameterDef(
                name="load_salt_mM",
                display_name="Load Salt",
                unit="mM",
                description="Salt concentration during loading",
                default=50.0,
                bounds=(0.0, 2000.0),
            ),
            ProcessParameterDef(
                name="wash_salt_mM",
                display_name="Wash Salt",
                unit="mM",
                description="Salt concentration during wash",
                default=50.0,
                bounds=(0.0, 2000.0),
            ),
            ProcessParameterDef(
                name="gradient_start_mM",
                display_name="Gradient Start",
                unit="mM",
                description="Salt concentration at gradient start",
                default=50.0,
                bounds=(0.0, 2000.0),
            ),
            ProcessParameterDef(
                name="gradient_end_mM",
                display_name="Gradient End",
                unit="mM",
                description="Salt concentration at gradient end",
                default=500.0,
                bounds=(0.0, 2000.0),
            ),
            ProcessParameterDef(
                name="strip_salt_mM",
                display_name="Strip Salt",
                unit="mM",
                description="Salt concentration during strip",
                default=1000.0,
                required=False,
                bounds=(0.0, 2000.0),
            ),
            # pH (optional)
            ProcessParameterDef(
                name="ph",
                display_name="pH",
                unit="-",
                description="Buffer pH",
                default=7.0,
                required=False,
                bounds=(2.0, 12.0),
            ),
        ]
    
    def get_component_experiment_parameters(self) -> list[ProcessParameterDef]:
        """Get per-component experiment parameters."""
        return [
            ProcessParameterDef(
                name="load_concentration",
                display_name="Load Concentration",
                unit="g/L",
                description="Concentration in the load (feed)",
                default=1.0,
                per_component=True,
                bounds=(0.0, 100.0),
            ),
        ]
    
    def create_process(
        self,
        experiment: ExperimentConfig,
        column_binding: ColumnBindingConfig,
    ) -> "Process":
        """Create a CADET Process from the configuration."""
        # Lazy imports for CADET-Process
        from CADETProcess.processModel import (
            ComponentSystem,
            FlowSheet,
            Inlet,
            Outlet,
            Process,
        )
        
        params = experiment.parameters
        components = experiment.components
        n_comp = len(components)
        
        # Build component system
        component_system = ComponentSystem()
        for comp in components:
            component_system.add_component(comp.name)
        
        # Build binding model
        binding_model = self._create_binding_model(
            column_binding.binding_model,
            component_system,
            column_binding.binding_parameters,
            column_binding.component_binding_parameters,
        )
        
        # Build column
        column = self._create_column(
            column_binding.column_model,
            component_system,
            binding_model,
            column_binding.column_parameters,
            column_binding.component_column_parameters,
        )
        
        # Calculate column volume (in m3)
        column_volume_m3 = column.volume
        column_volume_ml = column_volume_m3 * 1e6
        
        # Convert flow rate: mL/min -> m3/s
        flow_rate_ml_min = params["flow_rate_mL_min"]
        flow_rate_m3_s = flow_rate_ml_min / 1e6 / 60.0
        flow_rate_cv_min = flow_rate_ml_min / column_volume_ml
        
        # Convert volumes: CV -> seconds
        def cv_to_seconds(cv: float) -> float:
            if cv <= 0:
                return 0.0
            return cv / flow_rate_cv_min * 60.0
        
        equilibration_duration = cv_to_seconds(params.get("equilibration_cv", 0))
        load_duration = cv_to_seconds(params["load_cv"])
        wash_duration = cv_to_seconds(params["wash_cv"])
        elution_duration = cv_to_seconds(params["elution_cv"])
        strip_duration = cv_to_seconds(params.get("strip_cv", 0))
        
        # Salt concentrations
        load_salt = params["load_salt_mM"]
        wash_salt = params["wash_salt_mM"]
        gradient_end = params["gradient_end_mM"]
        strip_salt = params.get("strip_salt_mM", 1000.0)
        
        # Get load concentrations for each component
        load_concs = self._get_component_concentrations(params, components)
        
        # Build concentration arrays for each phase
        c_equilibration = [load_salt] + [0.0] * (n_comp - 1)
        c_load = [load_salt] + load_concs[1:]
        c_wash = [wash_salt] + [0.0] * (n_comp - 1)
        c_elution = [gradient_end] + [0.0] * (n_comp - 1)
        c_strip = [strip_salt] + [0.0] * (n_comp - 1)
        
        # Set initial column conditions
        column.c = c_equilibration
        if hasattr(column, 'cp'):
            column.cp = c_equilibration
        if hasattr(column, 'q') and hasattr(binding_model, 'capacity'):
            capacity = binding_model.capacity if hasattr(binding_model, 'capacity') else 0.0
            column.q = [capacity] + [0.0] * (n_comp - 1)
        
        # Build flow sheet
        flow_sheet = self._build_flow_sheet(column, component_system)
        
        # Create process
        process = Process(flow_sheet, experiment.name)
        
        # Calculate total cycle time
        cycle_time = (
            equilibration_duration +
            load_duration +
            wash_duration +
            elution_duration +
            strip_duration
        )
        process.cycle_time = cycle_time
        
        # Event times
        event_times = {
            'Load': equilibration_duration,
            'Wash': equilibration_duration + load_duration,
            'Elution': equilibration_duration + load_duration + wash_duration,
            'Strip': equilibration_duration + load_duration + wash_duration + elution_duration,
        }
        
        # Gradient slope
        gradient_slope = (np.array(c_elution) - np.array(c_wash)) / elution_duration
        c_gradient_poly = np.array(list(zip(c_wash, gradient_slope)))
        
        # Set up inlet
        inlet = flow_sheet.inlet
        inlet.flow_rate = flow_rate_m3_s
        
        # Add events
        process.add_event('Load', 'flow_sheet.inlet.c', c_load, time=event_times['Load'])
        process.add_event('Wash', 'flow_sheet.inlet.c', c_wash, time=event_times['Wash'])
        process.add_event('grad_start', 'flow_sheet.inlet.c', c_gradient_poly, event_times['Elution'])
        process.add_event('grad_stop', 'flow_sheet.inlet.c', c_strip, time=event_times['Strip'])
        
        return process
    
    def _get_component_concentrations(
        self,
        params: dict[str, Any],
        components: list,
    ) -> list[float]:
        """Extract load concentrations for each component."""
        concentrations = []
        
        for i, comp in enumerate(components):
            if comp.is_salt or i == 0:
                concentrations.append(params["load_salt_mM"])
            else:
                key = f"component_{i+1}_load_concentration"
                conc = params.get(key, 1.0)
                concentrations.append(conc)
        
        return concentrations
    
    def _build_flow_sheet(self, column, component_system) -> "FlowSheet":
        """Build the flow sheet with single inlet."""
        from CADETProcess.processModel import FlowSheet, Inlet, Outlet
        
        inlet = Inlet(component_system, name='inlet')
        outlet = Outlet(component_system, name='outlet')
        
        flow_sheet = FlowSheet(component_system)
        flow_sheet.add_unit(inlet, feed_inlet=True)
        flow_sheet.add_unit(column)
        flow_sheet.add_unit(outlet, product_outlet=True)
        
        flow_sheet.add_connection(inlet, column)
        flow_sheet.add_connection(column, outlet)
        
        return flow_sheet
