"""Load-Wash-Elute (LWE) operation mode with concentration-based gradient.

This operation mode uses a single inlet with concentration switching/gradients,
similar to how real chromatography systems work.
"""

from typing import Any, TYPE_CHECKING

# Lazy imports for CADET-Process (only needed when creating processes)
if TYPE_CHECKING:
    from CADETProcess.processModel import (
        ComponentSystem,
        FlowSheet,
        Inlet,
        Outlet,
        Process,
    )

from .base import (
    BaseOperationMode,
    ParameterDefinition,
    ParameterType,
    ExperimentConfig,
    ColumnBindingConfig,
    ComponentDefinition,
    SUPPORTED_COLUMN_MODELS,
    SUPPORTED_BINDING_MODELS,
)


class LWEConcentrationBased(BaseOperationMode):
    """Load-Wash-Elute process with concentration-based gradient elution.
    
    This mode uses a single inlet and modulates concentrations over time:
    - Equilibration: Column equilibrated at wash salt concentration
    - Load: Sample loaded at load salt concentration
    - Wash: Unbound material washed out at wash salt concentration
    - Elute: Linear salt gradient from wash to elution concentration
    - Strip (optional): High salt strip to clean column
    
    All volumes are specified in Column Volumes (CV) for lab convenience.
    Flow rate is specified in CV/min.
    Salt concentrations in mM.
    """
    
    name = "LWE_concentration_based"
    description = "Load-Wash-Elute with linear salt gradient elution"
    
    def get_experiment_parameters(self) -> list[ParameterDefinition]:
        """Get experiment parameters in lab-friendly units."""
        return [
            # Flow
            ParameterDefinition(
                name="flow_rate_cv_min",
                display_name="Flow Rate",
                unit="CV/min",
                description="Volumetric flow rate in column volumes per minute",
                default=1.0,
                bounds=(0.01, 10.0),
            ),
            # Volumes (all in CV)
            ParameterDefinition(
                name="equilibration_cv",
                display_name="Equilibration Volume",
                unit="CV",
                description="Volume for initial column equilibration",
                default=5.0,
                required=False,
                bounds=(0.0, 50.0),
            ),
            ParameterDefinition(
                name="load_cv",
                display_name="Load Volume",
                unit="CV",
                description="Sample loading volume",
                default=5.0,
                bounds=(0.1, 100.0),
            ),
            ParameterDefinition(
                name="wash_cv",
                display_name="Wash Volume",
                unit="CV",
                description="Post-load wash volume",
                default=5.0,
                bounds=(0.0, 50.0),
            ),
            ParameterDefinition(
                name="elution_cv",
                display_name="Elution Volume",
                unit="CV",
                description="Gradient elution volume",
                default=20.0,
                bounds=(1.0, 100.0),
            ),
            ParameterDefinition(
                name="strip_cv",
                display_name="Strip Volume",
                unit="CV",
                description="High salt strip volume (0 to skip)",
                default=0.0,
                required=False,
                bounds=(0.0, 20.0),
            ),
            # Salt concentrations
            ParameterDefinition(
                name="load_salt_mm",
                display_name="Load Salt",
                unit="mM",
                description="Salt concentration during loading",
                default=50.0,
                bounds=(0.0, 2000.0),
            ),
            ParameterDefinition(
                name="wash_salt_mm",
                display_name="Wash Salt",
                unit="mM",
                description="Salt concentration during wash",
                default=50.0,
                bounds=(0.0, 2000.0),
            ),
            ParameterDefinition(
                name="gradient_start_mm",
                display_name="Gradient Start",
                unit="mM",
                description="Salt concentration at gradient start",
                default=50.0,
                bounds=(0.0, 2000.0),
            ),
            ParameterDefinition(
                name="gradient_end_mm",
                display_name="Gradient End",
                unit="mM",
                description="Salt concentration at gradient end",
                default=500.0,
                bounds=(0.0, 2000.0),
            ),
            ParameterDefinition(
                name="strip_salt_mm",
                display_name="Strip Salt",
                unit="mM",
                description="Salt concentration during strip",
                default=1000.0,
                required=False,
                bounds=(0.0, 2000.0),
            ),
            # pH (optional, for record keeping / future GIEX support)
            ParameterDefinition(
                name="ph",
                display_name="pH",
                unit="-",
                description="Buffer pH",
                default=7.0,
                required=False,
                bounds=(2.0, 12.0),
            ),
        ]
    
    def get_component_experiment_parameters(self) -> list[ParameterDefinition]:
        """Get per-component experiment parameters."""
        return [
            ParameterDefinition(
                name="load_concentration",
                display_name="Load Concentration",
                unit="g/L",
                description="Concentration in the load (feed)",
                default=1.0,
                param_type=ParameterType.PER_COMPONENT,
                bounds=(0.0, 100.0),
            ),
        ]
    
    def create_process(
        self,
        experiment: ExperimentConfig,
        column_binding: ColumnBindingConfig,
    ) -> "Process":
        """Create a CADET Process from the configuration.
        
        Converts lab-friendly units to SI and builds the process.
        """
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
        
        # Calculate column volume (in m³)
        column_volume_m3 = column.volume  # CADET-Process calculates this
        column_volume_ml = column_volume_m3 * 1e6  # Convert to mL
        
        # Convert flow rate: CV/min -> m³/s
        flow_rate_cv_min = params.get("flow_rate_cv_min", 1.0)
        flow_rate_ml_min = flow_rate_cv_min * column_volume_ml
        flow_rate_m3_s = flow_rate_ml_min / 1e6 / 60.0  # mL/min -> m³/s
        
        # Convert volumes: CV -> seconds (duration = volume / flow_rate)
        def cv_to_seconds(cv: float) -> float:
            if cv <= 0:
                return 0.0
            return cv / flow_rate_cv_min * 60.0  # CV / (CV/min) * 60 s/min
        
        equilibration_duration = cv_to_seconds(params.get("equilibration_cv", 0.0))
        load_duration = cv_to_seconds(params.get("load_cv", 5.0))
        wash_duration = cv_to_seconds(params.get("wash_cv", 5.0))
        elution_duration = cv_to_seconds(params.get("elution_cv", 20.0))
        strip_duration = cv_to_seconds(params.get("strip_cv", 0.0))
        
        # Salt concentrations (already in mM which CADET uses)
        load_salt = params.get("load_salt_mm", 50.0)
        wash_salt = params.get("wash_salt_mm", 50.0)
        gradient_start = params.get("gradient_start_mm", 50.0)
        gradient_end = params.get("gradient_end_mm", 500.0)
        strip_salt = params.get("strip_salt_mm", 1000.0)
        
        # Build concentration arrays for each phase
        # Component 0 is salt, others are proteins
        
        # Get load concentrations for each component
        load_concs = self._get_component_concentrations(params, components)
        
        # Equilibration: salt at wash level, proteins at 0
        c_equilibration = [wash_salt] + [0.0] * (n_comp - 1)
        
        # Load: salt at load level, proteins at their concentrations
        c_load = [load_salt] + load_concs[1:]  # Skip salt in load_concs
        
        # Wash: salt at wash level, proteins at 0
        c_wash = [wash_salt] + [0.0] * (n_comp - 1)
        
        # Strip: high salt, no proteins
        c_strip = [strip_salt] + [0.0] * (n_comp - 1)
        
        # Set initial column conditions
        column.c = c_wash
        if hasattr(column, 'cp'):
            column.cp = c_wash
        if hasattr(column, 'q') and hasattr(binding_model, 'capacity'):
            # Salt occupies all binding sites initially
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
        
        # Set up inlet with constant flow rate
        inlet = flow_sheet.inlet
        inlet.flow_rate = flow_rate_m3_s
        
        # Track time for events
        t = 0.0
        
        # === Equilibration Phase ===
        if equilibration_duration > 0:
            process.add_event('equilibration', 'flow_sheet.inlet.c', c_equilibration, t)
            t += equilibration_duration
        
        # === Load Phase ===
        process.add_event('load', 'flow_sheet.inlet.c', c_load, t)
        t += load_duration
        
        # === Wash Phase ===
        process.add_event('wash', 'flow_sheet.inlet.c', c_wash, t)
        t += wash_duration
        
        # === Gradient Elution Phase ===
        # Linear gradient: salt goes from gradient_start to gradient_end
        if elution_duration > 0:
            gradient_slope = (gradient_end - gradient_start) / elution_duration
            
            # Polynomial coefficients: [constant, slope] for each component
            c_gradient = [[gradient_start, gradient_slope]]  # Salt: linear increase
            for _ in range(n_comp - 1):
                c_gradient.append([0.0, 0.0])  # Proteins: zero
            
            process.add_event('gradient_start', 'flow_sheet.inlet.c', c_gradient, t)
            t += elution_duration
            
            # End gradient (constant at gradient_end)
            c_gradient_end = [gradient_end] + [0.0] * (n_comp - 1)
            process.add_event('gradient_end', 'flow_sheet.inlet.c', c_gradient_end, t)
        
        # === Strip Phase ===
        if strip_duration > 0:
            process.add_event('strip', 'flow_sheet.inlet.c', c_strip, t)
        
        return process
    
    def _get_component_concentrations(
        self,
        params: dict[str, Any],
        components: list[ComponentDefinition],
    ) -> list[float]:
        """Extract load concentrations for each component from params.
        
        Looks for parameters like 'component_1_load_concentration'.
        """
        concentrations = []
        
        for i, comp in enumerate(components):
            # Salt component has concentration defined by load_salt_mm
            if comp.is_salt or i == 0:
                concentrations.append(params.get("load_salt_mm", 50.0))
            else:
                # Look for component-specific concentration
                key = f"component_{i+1}_load_concentration"
                conc = params.get(key, 0.0)
                concentrations.append(conc)
        
        return concentrations
    
    def _build_flow_sheet(
        self,
        column,
        component_system: "ComponentSystem",
    ) -> "FlowSheet":
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
    
    def _create_binding_model(
        self,
        binding_model_name: str,
        component_system: "ComponentSystem",
        scalar_params: dict[str, Any],
        component_params: dict[str, list[Any]],
    ):
        """Create and configure binding model."""
        class_path = SUPPORTED_BINDING_MODELS[binding_model_name]
        model_class = self._get_model_class(class_path)
        
        binding_model = model_class(component_system, name="binding")
        
        # Set scalar parameters
        for param, value in scalar_params.items():
            if hasattr(binding_model, param) and value is not None:
                setattr(binding_model, param, value)
        
        # Set per-component parameters
        for param, values in component_params.items():
            if hasattr(binding_model, param) and values:
                setattr(binding_model, param, values)
        
        return binding_model
    
    def _create_column(
        self,
        column_model_name: str,
        component_system: "ComponentSystem",
        binding_model,
        scalar_params: dict[str, Any],
        component_params: dict[str, list[Any]],
    ):
        """Create and configure column."""
        class_path = SUPPORTED_COLUMN_MODELS[column_model_name]
        model_class = self._get_model_class(class_path)
        
        column = model_class(component_system, name="column")
        column.binding_model = binding_model
        
        # Convert user units to SI for geometry
        # Length: cm -> m
        if "length" in scalar_params:
            length_cm = scalar_params["length"]
            column.length = length_cm / 100.0
        
        # Diameter: cm -> m (user enters in cm for convenience)
        if "diameter" in scalar_params:
            diameter_cm = scalar_params["diameter"]
            column.diameter = diameter_cm / 100.0
        
        # Particle radius: µm -> m
        if "particle_radius" in scalar_params and hasattr(column, 'particle_radius'):
            particle_radius_um = scalar_params["particle_radius"]
            column.particle_radius = particle_radius_um * 1e-6
        
        # Porosities (dimensionless, no conversion)
        if "bed_porosity" in scalar_params and hasattr(column, 'bed_porosity'):
            column.bed_porosity = scalar_params["bed_porosity"]
        
        if "particle_porosity" in scalar_params and hasattr(column, 'particle_porosity'):
            column.particle_porosity = scalar_params["particle_porosity"]
        
        if "total_porosity" in scalar_params and hasattr(column, 'total_porosity'):
            column.total_porosity = scalar_params["total_porosity"]
        
        # Axial dispersion (already in m²/s)
        if "axial_dispersion" in scalar_params:
            column.axial_dispersion = scalar_params["axial_dispersion"]
        
        # Set per-component parameters
        for param, values in component_params.items():
            if hasattr(column, param) and values:
                setattr(column, param, values)
        
        return column


# Convenience function to get the operation mode
def get_lwe_mode() -> LWEConcentrationBased:
    """Get an instance of the LWE operation mode."""
    return LWEConcentrationBased()
