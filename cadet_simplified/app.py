"""Simplified Panel GUI for CADET chromatography simulations.

A streamlined interface with Excel-based configuration:
1. Select models and components
2. Download template
3. Upload filled template
4. Validate and simulate
"""

import io
import json
from pathlib import Path

import panel as pn
import param
import pandas as pd
import numpy as np

pn.extension('tabulator', notifications=True)

from .operation_modes import (
    OPERATION_MODES,
    SUPPORTED_COLUMN_MODELS,
    SUPPORTED_BINDING_MODELS,
    get_operation_mode,
)
from .excel import ExcelTemplateGenerator, ExcelParser, ParseResult
from .storage import ExperimentStore, ExperimentSet
from .simulation import SimulationRunner, ValidationResult


class SimplifiedCADETApp(param.Parameterized):
    """Simplified CADET simulation application.
    
    Workflow:
    1. Configure: Select operation mode, models, components
    2. Template: Download Excel template
    3. Upload: Upload filled template, validate
    4. Simulate: Run simulations, view results
    """
    
    # Configuration parameters
    operation_mode = param.Selector(
        default="LWE_concentration_based",
        objects=list(OPERATION_MODES.keys()),
        doc="Operation mode (process type)",
    )
    column_model = param.Selector(
        default="LumpedRateModelWithPores",
        objects=list(SUPPORTED_COLUMN_MODELS.keys()),
        doc="Column model",
    )
    binding_model = param.Selector(
        default="StericMassAction",
        objects=list(SUPPORTED_BINDING_MODELS.keys()),
        doc="Binding model",
    )
    n_components = param.Integer(
        default=3,
        bounds=(2, 10),
        doc="Number of components (including salt)",
    )
    
    # Status
    status = param.String(default="Ready. Configure models and download template.")
    
    def __init__(
        self,
        storage_dir: str | Path = "./experiments",
        cadet_path: str | None = None,
        **params
    ):
        super().__init__(**params)
        
        self.storage_dir = Path(storage_dir)
        self.cadet_path = cadet_path
        
        # Initialize storage and runner
        self.store = ExperimentStore(self.storage_dir)
        self.runner = SimulationRunner(cadet_path)
        
        # State
        self._current_parse_result: ParseResult | None = None
        self._current_experiment_set: ExperimentSet | None = None
        self._simulation_results: list = []
        self._component_names: list[str] = self._default_component_names()
        
        # Build UI components
        self._build_ui()
    
    def _default_component_names(self) -> list[str]:
        """Generate default component names."""
        names = ["Salt"]
        for i in range(1, self.n_components):
            if i == 1:
                names.append("Product")
            else:
                names.append(f"Impurity{i-1}")
        return names
    
    def _build_ui(self):
        """Build UI components."""
        # === Tab 1: Configuration ===
        self._config_column = pn.Column(
            pn.pane.Markdown("## 1. Model Configuration"),
            pn.widgets.Select.from_param(self.param.operation_mode, name="Operation Mode"),
            pn.widgets.Select.from_param(self.param.column_model, name="Column Model"),
            pn.widgets.Select.from_param(self.param.binding_model, name="Binding Model"),
            pn.widgets.IntInput.from_param(self.param.n_components, name="Number of Components"),
            sizing_mode='stretch_width',
        )
        
        # Component names editor
        self._component_name_inputs = []
        self._component_names_column = pn.Column(
            pn.pane.Markdown("### Component Names"),
            sizing_mode='stretch_width',
        )
        self._update_component_name_inputs()
        
        # Download template button
        self._download_btn = pn.widgets.Button(
            name="Download Template",
            button_type="primary",
            width=200,
        )
        self._download_btn.on_click(self._on_download_template)
        
        self._template_download_area = pn.Column()
        
        # === Tab 2: Upload & Validate ===
        self._file_input = pn.widgets.FileInput(
            accept=".xlsx,.xls",
            name="Upload Filled Template",
        )
        self._file_input.param.watch(self._on_file_upload, 'value')
        
        self._validate_btn = pn.widgets.Button(
            name="Validate Configuration",
            button_type="warning",
            width=200,
            disabled=True,
        )
        self._validate_btn.on_click(self._on_validate)
        
        self._validation_output = pn.Column(
            pn.pane.Markdown("*Upload a filled template to validate*"),
            sizing_mode='stretch_width',
        )
        
        self._experiments_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            height=200,
            sizing_mode='stretch_width',
            disabled=True,
        )
        
        self._config_preview = pn.Column(
            pn.pane.Markdown("### Configuration Preview"),
            pn.pane.JSON({}, depth=2, name="Column/Binding Config"),
            sizing_mode='stretch_width',
        )
        
        # === Tab 3: Simulate ===
        self._simulate_btn = pn.widgets.Button(
            name="Run Simulations",
            button_type="success",
            width=200,
            disabled=True,
        )
        self._simulate_btn.on_click(self._on_simulate)
        
        self._simulation_progress = pn.indicators.Progress(
            name='Simulation Progress',
            value=0,
            max=100,
            sizing_mode='stretch_width',
            visible=False,
        )
        
        self._simulation_output = pn.Column(
            pn.pane.Markdown("*Validate configuration first, then run simulations*"),
            sizing_mode='stretch_width',
        )
        
        # === Tab 4: Results ===
        self._results_plot = pn.pane.Matplotlib(
            None,
            tight=True,
            sizing_mode='stretch_width',
            height=400,
        )
        
        self._experiment_selector = pn.widgets.Select(
            name="Select Experiment",
            options=[],
            disabled=True,
        )
        self._experiment_selector.param.watch(self._on_experiment_select, 'value')
        
        self._results_output = pn.Column(
            pn.pane.Markdown("*Run simulations to see results*"),
            sizing_mode='stretch_width',
        )
        
        # === Tab 5: Saved Experiments ===
        self._saved_experiments_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            height=300,
            sizing_mode='stretch_width',
        )
        self._refresh_saved_btn = pn.widgets.Button(
            name="Refresh",
            button_type="default",
            width=100,
        )
        self._refresh_saved_btn.on_click(self._on_refresh_saved)
        self._load_saved_btn = pn.widgets.Button(
            name="Load Selected",
            button_type="primary",
            width=150,
            disabled=True,
        )
        
        # Status bar
        self._status_pane = pn.pane.Alert(
            self.status,
            alert_type="info",
            sizing_mode='stretch_width',
        )
        
        # Watch for parameter changes
        self.param.watch(self._on_n_components_change, 'n_components')
    
    def _update_component_name_inputs(self):
        """Update component name input widgets."""
        self._component_name_inputs = []
        items = [pn.pane.Markdown("### Component Names")]
        
        for i in range(self.n_components):
            default_name = self._component_names[i] if i < len(self._component_names) else f"Component_{i}"
            inp = pn.widgets.TextInput(
                name=f"Component {i+1}",
                value=default_name,
                width=200,
            )
            self._component_name_inputs.append(inp)
            items.append(inp)
        
        self._component_names_column.objects = items
    
    def _on_n_components_change(self, event):
        """Handle change in number of components."""
        # Update default names
        while len(self._component_names) < self.n_components:
            idx = len(self._component_names)
            if idx == 0:
                self._component_names.append("Salt")
            elif idx == 1:
                self._component_names.append("Product")
            else:
                self._component_names.append(f"Impurity{idx-1}")
        
        self._component_names = self._component_names[:self.n_components]
        self._update_component_name_inputs()
    
    def _get_component_names(self) -> list[str]:
        """Get component names from input widgets."""
        return [inp.value for inp in self._component_name_inputs]
    
    def _on_download_template(self, event):
        """Generate and provide template for download."""
        try:
            mode = get_operation_mode(self.operation_mode)
            component_names = self._get_component_names()
            
            generator = ExcelTemplateGenerator(
                operation_mode=mode,
                column_model=self.column_model,
                binding_model=self.binding_model,
                n_components=self.n_components,
                component_names=component_names,
            )
            
            # Generate bytes
            template_bytes = generator.to_bytes()
            
            # Create download widget
            filename = f"template_{self.operation_mode}_{self.n_components}comp.xlsx"
            
            download_widget = pn.widgets.FileDownload(
                io.BytesIO(template_bytes),
                filename=filename,
                button_type="success",
                label="Click to Download Template",
            )
            
            self._template_download_area.objects = [
                pn.pane.Markdown(f"✓ Template generated: **{filename}**"),
                download_widget,
            ]
            
            self.status = f"Template generated. Download and fill in your experiments."
            self._update_status("success")
            
        except Exception as e:
            self.status = f"Error generating template: {str(e)}"
            self._update_status("danger")
    
    def _on_file_upload(self, event):
        """Handle file upload."""
        if event.new is None:
            return
        
        try:
            # Parse the uploaded file
            file_bytes = io.BytesIO(event.new)
            parser = ExcelParser()
            result = parser.parse(file_bytes)
            
            self._current_parse_result = result
            
            if not result.success:
                # Show errors
                error_md = "### ❌ Parse Errors\n\n"
                for error in result.errors:
                    error_md += f"- {error}\n"
                if result.warnings:
                    error_md += "\n### ⚠️ Warnings\n\n"
                    for warning in result.warnings:
                        error_md += f"- {warning}\n"
                
                self._validation_output.objects = [pn.pane.Markdown(error_md)]
                self._validate_btn.disabled = True
                self.status = "Parse errors in uploaded file. Fix and re-upload."
                self._update_status("danger")
                return
            
            # Show parsed experiments
            exp_data = []
            for exp in result.experiments:
                row = {"name": exp.name}
                # Add key parameters
                # TODO remove hardcoding
                for key in ["flow_rate_cv_min", "load_cv", "wash_cv", "elution_cv", 
                           "gradient_start_mM", "gradient_end_mM"]:
                    if key in exp.parameters:
                        row[key] = exp.parameters[key]
                exp_data.append(row)
            
            df = pd.DataFrame(exp_data)
            self._experiments_table.value = df
            
            # Show column/binding config preview
            if result.column_binding:
                config_dict = {
                    "column_model": result.column_binding.column_model,
                    "binding_model": result.column_binding.binding_model,
                    "column_parameters": result.column_binding.column_parameters,
                    "binding_parameters": result.column_binding.binding_parameters,
                }
                self._config_preview.objects = [
                    pn.pane.Markdown("### Configuration Preview"),
                    pn.pane.JSON(config_dict, depth=3, name="Config"),
                ]
            
            # Enable validation
            self._validate_btn.disabled = False
            
            n_exp = len(result.experiments)
            self.status = f"Parsed {n_exp} experiment(s). Click 'Validate Configuration' to check."
            self._update_status("info")
            
            # Show success message
            success_md = f"### ✓ Parsed Successfully\n\n"
            success_md += f"- **{n_exp}** experiments found\n"
            success_md += f"- Column model: {result.column_binding.column_model}\n"
            success_md += f"- Binding model: {result.column_binding.binding_model}\n"
            
            if result.warnings:
                success_md += "\n### ⚠️ Warnings\n\n"
                for warning in result.warnings:
                    success_md += f"- {warning}\n"
            
            self._validation_output.objects = [pn.pane.Markdown(success_md)]
            
        except Exception as e:
            self.status = f"Error reading file: {str(e)}"
            self._update_status("danger")
            self._validation_output.objects = [
                pn.pane.Markdown(f"### ❌ Error\n\n{str(e)}")
            ]
    
    def _on_validate(self, event):
        """Validate the uploaded configuration."""
        if self._current_parse_result is None:
            return
        
        result = self._current_parse_result
        mode = get_operation_mode(self.operation_mode)
        
        validation_results = []
        all_valid = True
        
        for exp in result.experiments:
            try:
                # Try to create the process
                process = mode.create_process(exp, result.column_binding)
                
                # Validate with CADET-Process
                val_result = self.runner.validate(process, exp.name)
                validation_results.append(val_result)
                
                if not val_result.valid:
                    all_valid = False
                    
            except Exception as e:
                validation_results.append(ValidationResult(
                    experiment_name=exp.name,
                    valid=False,
                    errors=[f"Failed to create process: {str(e)}"],
                ))
                all_valid = False
        
        # Display validation results
        if all_valid:
            md = "### ✓ All Configurations Valid\n\n"
            for vr in validation_results:
                md += f"- **{vr.experiment_name}**: ✓ Valid\n"
                if vr.warnings:
                    for w in vr.warnings:
                        md += f"  - ⚠️ {w}\n"
            
            self._simulate_btn.disabled = False
            self.status = "Configuration valid! Ready to simulate."
            self._update_status("success")
        else:
            md = "### ❌ Validation Errors\n\n"
            for vr in validation_results:
                if vr.valid:
                    md += f"- **{vr.experiment_name}**: ✓ Valid\n"
                else:
                    md += f"- **{vr.experiment_name}**: ❌ Invalid\n"
                    for err in vr.errors:
                        md += f"  - {err}\n"
            
            self._simulate_btn.disabled = True
            self.status = "Validation failed. Fix errors and re-upload."
            self._update_status("danger")
        
        self._validation_output.objects = [
            self._validation_output.objects[0] if self._validation_output.objects else pn.pane.Markdown(""),
            pn.pane.Markdown(md),
        ]
    
    def _on_simulate(self, event):
        """Run simulations."""
        if self._current_parse_result is None:
            return
        
        result = self._current_parse_result
        mode = get_operation_mode(self.operation_mode)
        
        self._simulation_results = []
        self._simulation_progress.visible = True
        self._simulation_progress.value = 0
        
        n_experiments = len(result.experiments)
        
        output_items = [pn.pane.Markdown("### Simulation Progress\n")]
        
        for i, exp in enumerate(result.experiments):
            # Update progress
            progress = int((i / n_experiments) * 100)
            self._simulation_progress.value = progress
            self.status = f"Simulating {exp.name} ({i+1}/{n_experiments})..."
            self._update_status("info")
            
            try:
                # Create process
                process = mode.create_process(exp, result.column_binding)
                
                # Run simulation
                sim_result = self.runner.run(process, exp.name)
                self._simulation_results.append(sim_result)
                
                if sim_result.success:
                    output_items.append(
                        pn.pane.Markdown(f"✓ **{exp.name}**: Completed in {sim_result.runtime_seconds:.2f}s")
                    )
                else:
                    error_msg = "; ".join(sim_result.errors[:2])  # Show first 2 errors
                    output_items.append(
                        pn.pane.Markdown(f"❌ **{exp.name}**: Failed - {error_msg}")
                    )
                    
            except Exception as e:
                output_items.append(
                    pn.pane.Markdown(f"❌ **{exp.name}**: Error - {str(e)}")
                )
        
        self._simulation_progress.value = 100
        self._simulation_progress.visible = False
        
        # Update results
        successful = sum(1 for r in self._simulation_results if r.success)
        self.status = f"Completed: {successful}/{n_experiments} simulations successful."
        self._update_status("success" if successful == n_experiments else "warning")
        
        self._simulation_output.objects = output_items
        
        # Enable results viewing
        if successful > 0:
            exp_names = [r.experiment_name for r in self._simulation_results if r.success]
            self._experiment_selector.options = exp_names
            self._experiment_selector.disabled = False
            if exp_names:
                self._experiment_selector.value = exp_names[0]
                self._update_results_plot(exp_names[0])
        
        # Save experiment set
        if successful > 0:
            try:
                self._current_experiment_set = self.store.save_from_parse_result(
                    experiments=result.experiments,
                    column_binding=result.column_binding,
                    name=f"Simulation_{len(self.store.list_all())+1}",
                    operation_mode=self.operation_mode,
                )
            except Exception as e:
                print(f"Warning: Could not save experiment set: {e}")
    
    def _on_experiment_select(self, event):
        """Handle experiment selection for results viewing."""
        if event.new:
            self._update_results_plot(event.new)
    
    def _update_results_plot(self, experiment_name: str):
        """Update the results plot for selected experiment."""
        # Find the result
        result = None
        for r in self._simulation_results:
            if r.experiment_name == experiment_name:
                result = r
                break
        
        if result is None or not result.success:
            self._results_plot.object = None
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            time = result.time
            
            # Plot each component
            for comp_name, conc in result.solution.items():
                ax.plot(time / 60, conc, label=comp_name)  # Convert time to minutes
            
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Concentration (mM)")
            ax.set_title(f"Chromatogram: {experiment_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self._results_plot.object = fig
            
        except Exception as e:
            self._results_output.objects = [
                pn.pane.Markdown(f"Error creating plot: {str(e)}")
            ]
    
    def _on_refresh_saved(self, event):
        """Refresh saved experiments list."""
        saved = self.store.list_all()
        if saved:
            df = pd.DataFrame(saved)
            self._saved_experiments_table.value = df
        else:
            self._saved_experiments_table.value = pd.DataFrame()
    
    def _update_status(self, alert_type: str):
        """Update status pane."""
        self._status_pane.alert_type = alert_type
        self._status_pane.object = self.status
    
    def view(self) -> pn.viewable:
        """Create the main application view."""
        # Tab 1: Configuration
        config_tab = pn.Column(
            self._config_column,
            self._component_names_column,
            pn.Row(self._download_btn),
            self._template_download_area,
            sizing_mode='stretch_width',
        )
        
        # Tab 2: Upload & Validate
        upload_tab = pn.Column(
            pn.pane.Markdown("## 2. Upload Filled Template"),
            self._file_input,
            pn.pane.Markdown("### Parsed Experiments"),
            self._experiments_table,
            self._config_preview,
            pn.Row(self._validate_btn),
            self._validation_output,
            sizing_mode='stretch_width',
        )
        
        # Tab 3: Simulate
        simulate_tab = pn.Column(
            pn.pane.Markdown("## 3. Run Simulations"),
            pn.Row(self._simulate_btn),
            self._simulation_progress,
            self._simulation_output,
            sizing_mode='stretch_width',
        )
        
        # Tab 4: Results
        results_tab = pn.Column(
            pn.pane.Markdown("## 4. View Results"),
            self._experiment_selector,
            self._results_plot,
            self._results_output,
            sizing_mode='stretch_width',
        )
        
        # Tab 5: Saved Experiments
        saved_tab = pn.Column(
            pn.pane.Markdown("## Saved Experiment Sets"),
            pn.Row(self._refresh_saved_btn, self._load_saved_btn),
            self._saved_experiments_table,
            sizing_mode='stretch_width',
        )
        
        # Assemble tabs
        tabs = pn.Tabs(
            ("1. Configure", config_tab),
            ("2. Upload", upload_tab),
            ("3. Simulate", simulate_tab),
            ("4. Results", results_tab),
            ("5. Saved", saved_tab),
            sizing_mode='stretch_width',
        )
        
        # Main layout
        layout = pn.Column(
            pn.pane.Markdown("# CADET Simplified"),
            self._status_pane,
            tabs,
            sizing_mode='stretch_width',
        )
        
        return layout


def create_app(storage_dir: str = "./experiments", cadet_path: str | None = None):
    """Create the application.
    
    Parameters
    ----------
    storage_dir : str
        Directory for storing experiment data
    cadet_path : str, optional
        Path to CADET installation
        
    Returns
    -------
    SimplifiedCADETApp
        The application instance
    """
    return SimplifiedCADETApp(storage_dir=storage_dir, cadet_path=cadet_path)


def serve(storage_dir: str = "./experiments", cadet_path: str | None = None, **kwargs):
    """Serve the application.
    
    Parameters
    ----------
    storage_dir : str
        Directory for storing experiment data
    cadet_path : str, optional
        Path to CADET installation
    **kwargs
        Additional arguments passed to pn.serve()
    """
    app = create_app(storage_dir=storage_dir, cadet_path=cadet_path)
    pn.serve(app.view(), **kwargs)


# For running directly
if __name__ == "__main__":
    serve(port=5007, show=True)
