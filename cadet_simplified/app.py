"""Simplified Panel GUI for CADET chromatography simulations.

A streamlined interface with Excel-based configuration:
1. Select models and components
2. Download template
3. Upload filled template
4. Validate and simulate
5. Browse saved experiments
6. Analyse selected experiments

Run with:
    panel serve app.py --show --autoreload
    
Or programmatically:
    from cadet_simplified.app import serve
    serve(port=5006)
"""

import io
from pathlib import Path
from datetime import datetime

import panel as pn
import param
import pandas as pd

pn.extension('tabulator', notifications=True)

# Import from refactored modules
from .configs import list_binding_models, list_column_models
from .operation_modes import get_operation_mode, list_operation_modes
from .excel import ExcelTemplateGenerator, parse_excel, ParseResult
from .storage import FileStorage, LoadedExperiment
from .simulation import SimulationRunner, SimulationResultWrapper, ValidationResult
from .plotting import (
    plot_chromatogram_from_df,
    plot_chromatogram_overlay_from_df,
    plot_inlet_profile,
    calculate_column_volume_mL,
)


# Maximum number of chromatograms to preview in Saved tab
MAX_PREVIEW_CHROMATOGRAMS = 10


class SimplifiedCADETApp(param.Parameterized):
    """Simplified CADET simulation application.
    
    Workflow:
    1. Configure: Select operation mode, models, components
    2. Template: Download Excel template
    3. Upload: Upload filled template, validate
    4. Simulate: Run simulations, view results
    5. Saved: Browse and select saved experiments
    6. Analysis: Analyze selected experiments
    """
    
    # Configuration parameters
    operation_mode = param.Selector(
        default="LWE_concentration_based",
        objects=list_operation_modes(),
        doc="Operation mode (process type)",
    )
    column_model = param.Selector(
        default="LumpedRateModelWithPores",
        objects=list_column_models(),
        doc="Column model",
    )
    binding_model = param.Selector(
        default="StericMassAction",
        objects=list_binding_models(),
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
        self.storage = FileStorage(self.storage_dir)
        self.runner = SimulationRunner(cadet_path)
        
        # State
        self._current_parse_result: ParseResult | None = None
        self._simulation_results: list[SimulationResultWrapper] = []
        self._component_names: list[str] = self._default_component_names()
        self._validated_processes: list = []  # Store validated processes for simulation
        
        # Analysis state
        self._loaded_experiments: list[LoadedExperiment] = []
        
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
        
        # Checkbox for normalized inlet plot
        self._inlet_normalized_checkbox = pn.widgets.Checkbox(
            name="Normalize inlet plot",
            value=False,
        )
        
        # Dropdown for inlet plot x-axis
        self._inlet_x_axis_select = pn.widgets.Select(
            name="X-Axis",
            options={"Time (min)": "minutes", "Column Volumes (CV)": "cv"},
            value="minutes",
            width=180,
        )
        
        self._validation_output = pn.Column(
            pn.pane.Markdown("*Upload a filled template to validate*"),
            sizing_mode='stretch_width',
        )
        
        # Area for inlet profile plot (shown after validation)
        self._inlet_plot_area = pn.Column(
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
        
        self._simulation_status_text = pn.pane.Markdown(
            "",
            sizing_mode='stretch_width',
        )
        
        self._simulation_output = pn.Column(
            pn.pane.Markdown("*Validate configuration first, then run simulations*"),
            sizing_mode='stretch_width',
        )
        
        # === Tab 4: Saved Experiments ===
        self._saved_experiments_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            height=300,
            sizing_mode='stretch_width',
            selectable='checkbox',
            pagination='local',
            page_size=25,
        )
        
        self._refresh_saved_btn = pn.widgets.Button(
            name="Refresh",
            button_type="default",
            width=100,
        )
        self._refresh_saved_btn.on_click(self._on_refresh_saved)
        
        self._load_data_btn = pn.widgets.Button(
            name="Load Selected for Analysis",
            button_type="primary",
            width=200,
        )
        self._load_data_btn.on_click(self._on_load_data)
        
        # Preview Selected button and plot area
        self._preview_selected_btn = pn.widgets.Button(
            name="Preview Selected",
            button_type="light",
            width=150,
        )
        self._preview_selected_btn.on_click(self._on_preview_selected)
        
        self._preview_plot_area = pn.Column(
            sizing_mode='stretch_width',
        )
        
        self._load_progress = pn.indicators.Progress(
            name='Loading',
            value=0,
            max=100,
            sizing_mode='stretch_width',
            visible=False,
        )
        
        self._loaded_info = pn.pane.Markdown(
            "*Select experiments and click 'Load Selected for Analysis'*"
        )
        
        # === Tab 5: Analysis ===
        self._analysis_type_selector = pn.widgets.Select(
            name="Analysis Type",
            options={
                "Chromatogram Overlay": "overlay",
                "Individual Chromatograms": "individual",
            },
            value="overlay",
            width=200,
        )
        
        # Analysis plot options
        self._analysis_normalized_checkbox = pn.widgets.Checkbox(
            name="Normalize",
            value=False,
        )
        
        self._analysis_x_axis_select = pn.widgets.Select(
            name="X-Axis",
            options={"Time (min)": "minutes", "Column Volumes (CV)": "cv"},
            value="minutes",
            width=180,
        )
        
        self._analyse_btn = pn.widgets.Button(
            name="Run Analysis",
            button_type="success",
            width=150,
            disabled=True,
        )
        self._analyse_btn.on_click(self._on_analyse)
        
        self._analysis_container = pn.Column(
            pn.pane.Markdown("*Load experiments from the 'Saved' tab first*"),
            sizing_mode='stretch_width',
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
            component_names = self._get_component_names()
            
            generator = ExcelTemplateGenerator(
                operation_mode=self.operation_mode,
                column_model=self.column_model,
                binding_model=self.binding_model,
                n_components=self.n_components,
                component_names=component_names,
            )
            
            template_bytes = generator.to_bytes()
            filename = f"template_{self.operation_mode}_{self.n_components}comp.xlsx"
            
            download_widget = pn.widgets.FileDownload(
                io.BytesIO(template_bytes),
                filename=filename,
                button_type="success",
                label="Click to Download Template",
            )
            
            self._template_download_area.objects = [
                pn.pane.Alert(f"Template generated: {filename}", alert_type="success"),
                download_widget,
            ]
            
            self.status = "Template generated. Download and fill in your experiments."
            self._update_status("success")
            
        except Exception as e:
            self.status = f"Error generating template: {str(e)}"
            self._update_status("danger")
    
    def _on_file_upload(self, event):
        """Handle file upload."""
        if event.new is None:
            return
        
        try:
            file_bytes = io.BytesIO(event.new)
            result = parse_excel(file_bytes)
            
            self._current_parse_result = result
            
            if not result.success:
                error_md = "### Parse Errors\n\n"
                for error in result.errors:
                    error_md += f"- {error}\n"
                if result.warnings:
                    error_md += "\n### Warnings\n\n"
                    for warning in result.warnings:
                        error_md += f"- {warning}\n"
                
                self._validation_output.objects = [
                    pn.pane.Alert("Parse errors in uploaded file", alert_type="danger"),
                    pn.pane.Markdown(error_md),
                ]
                self._validate_btn.disabled = True
                self._inlet_plot_area.objects = []
                self.status = "Parse errors in uploaded file. Fix and re-upload."
                self._update_status("danger")
                return
            
            # Show parsed experiments
            exp_data = []
            for exp in result.experiments:
                row = {"name": exp.name}
                for key in ["flow_rate_mL_min", "load_cv", "wash_cv", "elution_cv", 
                           "gradient_start_mM", "gradient_end_mM"]:
                    if key in exp.parameters:
                        row[key] = exp.parameters[key]
                exp_data.append(row)
            
            df = pd.DataFrame(exp_data)
            self._experiments_table.value = df
            
            # Show config preview
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
            
            self._validate_btn.disabled = False
            self._inlet_plot_area.objects = []
            
            n_exp = len(result.experiments)
            self.status = f"Parsed {n_exp} experiment(s). Click 'Validate Configuration' to check."
            self._update_status("info")
            
            success_md = f"### Parse Summary\n\n"
            success_md += f"- **{n_exp}** experiments found\n"
            success_md += f"- Column model: {result.column_binding.column_model}\n"
            success_md += f"- Binding model: {result.column_binding.binding_model}\n"
            
            if result.warnings:
                success_md += "\n### Warnings\n\n"
                for warning in result.warnings:
                    success_md += f"- {warning}\n"
            
            self._validation_output.objects = [
                pn.pane.Alert("File parsed successfully", alert_type="success"),
                pn.pane.Markdown(success_md),
            ]
            
        except Exception as e:
            self.status = f"Error reading file: {str(e)}"
            self._update_status("danger")
            self._validation_output.objects = [
                pn.pane.Alert(f"Error reading file: {str(e)}", alert_type="danger"),
            ]
    
    def _on_validate(self, event):
        """Validate the uploaded configuration."""
        if self._current_parse_result is None:
            return
        
        result = self._current_parse_result
        mode = get_operation_mode(self.operation_mode)
        
        validation_results = []
        all_valid = True
        self._validated_processes = []
        
        for exp in result.experiments:
            try:
                process = mode.create_process(exp, result.column_binding)
                val_result = self.runner.validate(process, exp.name)
                validation_results.append(val_result)
                
                if val_result.valid:
                    self._validated_processes.append(process)
                else:
                    all_valid = False
                    self._validated_processes.append(None)
                    
            except Exception as e:
                validation_results.append(ValidationResult(
                    experiment_name=exp.name,
                    valid=False,
                    errors=[f"Failed to create process: {str(e)}"],
                ))
                all_valid = False
                self._validated_processes.append(None)
        
        if all_valid:
            md = "### Validation Results\n\n"
            for vr in validation_results:
                md += f"- **{vr.experiment_name}**: Valid\n"
                if vr.warnings:
                    for w in vr.warnings:
                        md += f"  - Warning: {w}\n"
            
            self._simulate_btn.disabled = False
            self.status = "Configuration valid! Ready to simulate."
            self._update_status("success")
            
            # Show inlet profile plot for first valid process
            self._show_inlet_profile_plot()
            
            # Update validation output with Alert + details
            self._validation_output.objects = [
                self._validation_output.objects[0] if self._validation_output.objects else pn.pane.Markdown(""),
                pn.pane.Alert("All configurations valid", alert_type="success"),
                pn.pane.Markdown(md),
            ]
            
        else:
            md = "### Validation Results\n\n"
            for vr in validation_results:
                if vr.valid:
                    md += f"- **{vr.experiment_name}**: Valid\n"
                else:
                    md += f"- **{vr.experiment_name}**: FAILED\n"
                    for err in vr.errors:
                        md += f"  - {err}\n"
            
            self._simulate_btn.disabled = True
            self._inlet_plot_area.objects = []
            self.status = "Validation failed. Fix errors and re-upload."
            self._update_status("danger")
            
            # Update validation output with Alert + details
            self._validation_output.objects = [
                self._validation_output.objects[0] if self._validation_output.objects else pn.pane.Markdown(""),
                pn.pane.Alert("Validation errors found", alert_type="danger"),
                pn.pane.Markdown(md),
            ]
    
    def _show_inlet_profile_plot(self):
        """Show inlet profile plot for the first validated process."""
        # Find first valid process
        process = None
        for p in self._validated_processes:
            if p is not None:
                process = p
                break
        
        if process is None:
            self._inlet_plot_area.objects = []
            return
        
        try:
            normalized = self._inlet_normalized_checkbox.value
            x_axis = self._inlet_x_axis_select.value
            
            plot = plot_inlet_profile(
                process,
                title=f"Inlet Profile: {process.name}",
                normalized=normalized,
                x_axis=x_axis,
                width=700,
                height=350,
            )
            
            # Watch widgets for changes
            def update_plot(event):
                self._show_inlet_profile_plot()
            
            # Only set up watchers once (check if already watching)
            if not hasattr(self, '_inlet_watchers_set'):
                self._inlet_normalized_checkbox.param.watch(update_plot, 'value')
                self._inlet_x_axis_select.param.watch(update_plot, 'value')
                self._inlet_watchers_set = True
            
            self._inlet_plot_area.objects = [
                pn.pane.Markdown("### Inlet Concentration Profile"),
                pn.Row(
                    self._inlet_normalized_checkbox,
                    self._inlet_x_axis_select,
                ),
                pn.pane.HoloViews(plot, sizing_mode='stretch_width'),
            ]
            
        except Exception as e:
            self._inlet_plot_area.objects = [
                pn.pane.Alert(f"Could not generate inlet plot: {e}", alert_type="warning")
            ]
    
    def _on_simulate(self, event):
        """Run simulations using run_batch with progress callback."""
        if self._current_parse_result is None:
            return
        
        if not self._validated_processes:
            self.status = "No validated processes. Please validate first."
            self._update_status("warning")
            return
        
        result = self._current_parse_result
        
        # Filter to only valid processes
        valid_processes = [p for p in self._validated_processes if p is not None]
        
        if not valid_processes:
            self.status = "No valid processes to simulate."
            self._update_status("warning")
            return
        
        self._simulation_progress.visible = True
        self._simulation_progress.value = 0
        self._simulation_status_text.object = ""
        
        output_items = [pn.pane.Markdown("### Simulation Progress\n")]
        self._simulation_output.objects = output_items
        
        n_total = len(valid_processes)
        
        # Progress callback for run_batch
        def progress_callback(current, total, sim_result):
            progress = int((current / total) * 100)
            self._simulation_progress.value = progress
            self._simulation_status_text.object = f"**Simulating {current}/{total}...**"
            
            # Update output items
            if sim_result.success:
                output_items.append(
                    pn.pane.Alert(
                        f"{sim_result.experiment_name}: Completed in {sim_result.runtime_seconds:.2f}s",
                        alert_type="success"
                    )
                )
            else:
                error_msg = "; ".join(sim_result.errors[:2]) if sim_result.errors else "Unknown error"
                output_items.append(
                    pn.pane.Alert(
                        f"{sim_result.experiment_name}: Failed - {error_msg}",
                        alert_type="danger"
                    )
                )
            self._simulation_output.objects = output_items
        
        # Run batch simulation
        self._simulation_results = self.runner.run_batch(
            valid_processes,
            progress_callback=progress_callback,
        )
        
        self._simulation_progress.value = 100
        self._simulation_progress.visible = False
        self._simulation_status_text.object = ""
        
        successful = sum(1 for r in self._simulation_results if r.success)
        self.status = f"Completed: {successful}/{n_total} simulations successful."
        self._update_status("success" if successful == n_total else "warning")
        
        # Save to storage if any successful
        if successful > 0:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                set_name = f"Simulation_{timestamp}"
                
                # Match results back to experiments
                valid_experiments = [
                    exp for exp, p in zip(result.experiments, self._validated_processes)
                    if p is not None
                ]
                
                set_id = self.storage.save_experiment_set(
                    name=set_name,
                    operation_mode=self.operation_mode,
                    experiments=valid_experiments,
                    column_binding=result.column_binding,
                    results=self._simulation_results,
                )
                
                output_items.append(pn.pane.Alert(
                    f"Results saved. Set ID: {set_id}",
                    alert_type="success"
                ))
                output_items.append(pn.pane.Markdown("*Go to the 'Saved' tab to browse and analyze results.*"))
                self._simulation_output.objects = output_items
                
            except Exception as e:
                output_items.append(pn.pane.Alert(
                    f"Could not save results: {e}",
                    alert_type="warning"
                ))
                self._simulation_output.objects = output_items
    
    def _on_refresh_saved(self, event):
        """Refresh saved experiments table."""
        try:
            df = self.storage.list_experiments(limit=50)
            
            if df.empty:
                self._saved_experiments_table.value = pd.DataFrame({
                    "experiment_set_name": [],
                    "experiment_name": [],
                    "created_at": [],
                    "n_components": [],
                    "column_model": [],
                    "binding_model": [],
                })
                self._loaded_info.object = "*No saved experiments found. Run some simulations first.*"
            else:
                # Select columns for display
                display_cols = [
                    "experiment_set_id", "experiment_set_name", "experiment_name",
                    "created_at", "n_components", "column_model", "binding_model",
                    "has_results",
                ]
                display_df = df[[c for c in display_cols if c in df.columns]].copy()
                self._saved_experiments_table.value = display_df
                self._loaded_info.object = f"*{len(df)} experiment(s) available. Select and click 'Load Selected for Analysis'.*"
            
            # Clear preview
            self._preview_plot_area.objects = []
                
        except Exception as e:
            self._loaded_info.object = f"*Error loading experiments: {e}*"
    
    def _on_preview_selected(self, event):
        """Preview chromatograms for selected experiments (quick view)."""
        selection = self._saved_experiments_table.selection
        
        if not selection:
            self._preview_plot_area.objects = [
                pn.pane.Alert("No experiments selected. Click checkboxes to select.", alert_type="info")
            ]
            return
        
        df = self._saved_experiments_table.value
        if df is None or df.empty:
            return
        
        # Get selected rows
        selected_rows = df.iloc[selection]
        n_selected = len(selected_rows)
        
        # Warn and limit if too many
        if n_selected > MAX_PREVIEW_CHROMATOGRAMS:
            show_limit_warning = True
            selected_rows = selected_rows.head(MAX_PREVIEW_CHROMATOGRAMS)
        else:
            show_limit_warning = False
        
        # Load chromatograms (fast path - parquet only)
        chromatograms = []
        for _, row in selected_rows.iterrows():
            set_id = row["experiment_set_id"]
            exp_name = row["experiment_name"]
            set_name = row.get("experiment_set_name", set_id)
            
            chrom_df = self.storage.get_chromatogram(set_id, exp_name)
            if chrom_df is not None:
                label = f"{set_name}/{exp_name}"
                chromatograms.append((label, chrom_df))
        
        if not chromatograms:
            self._preview_plot_area.objects = [
                pn.pane.Alert("No chromatogram data available for selected experiments.", alert_type="warning")
            ]
            return
        
        # Create overlay plot (preview uses minutes, no CV option)
        try:
            plot = plot_chromatogram_overlay_from_df(
                chromatograms,
                title=f"Preview ({len(chromatograms)} experiments)",
                width=800,
                height=400,
            )
            
            components = []
            if show_limit_warning:
                components.append(pn.pane.Alert(
                    f"{n_selected} experiments selected. Showing first {MAX_PREVIEW_CHROMATOGRAMS} only.",
                    alert_type="warning"
                ))
            components.append(pn.pane.HoloViews(plot, sizing_mode='stretch_width'))
            
            self._preview_plot_area.objects = components
            
        except Exception as e:
            self._preview_plot_area.objects = [
                pn.pane.Alert(f"Could not create preview: {e}", alert_type="danger")
            ]
    
    def _on_load_data(self, event):
        """Load selected experiments for analysis."""
        selection = self._saved_experiments_table.selection
        
        if not selection:
            self._loaded_info.object = "*No experiments selected. Click checkboxes to select.*"
            return
        
        df = self._saved_experiments_table.value
        if df is None or df.empty:
            return
        
        # Get selected rows
        selected_rows = df.iloc[selection]
        
        # Build selection list
        selections = [
            (row["experiment_set_id"], row["experiment_name"])
            for _, row in selected_rows.iterrows()
        ]
        
        self._load_progress.visible = True
        self._load_progress.value = 50
        self._loaded_info.object = f"*Loading {len(selections)} experiment(s)...*"
        
        try:
            self._loaded_experiments = self.storage.load_results_by_selection(
                selections=selections,
                include_chromatogram=True,
            )
            
            self._load_progress.value = 100
            self._load_progress.visible = False
            
            n_loaded = len(self._loaded_experiments)
            self._loaded_info.object = f"**{n_loaded}** experiment(s) loaded. Go to 'Analysis' tab to analyze."
            
            # Enable analysis button
            self._analyse_btn.disabled = False
            
            # Update analysis tab placeholder
            self._analysis_container.objects = [
                pn.pane.Markdown(f"**{n_loaded} experiment(s) loaded.** Select analysis type and click 'Run Analysis'."),
            ]
            
        except Exception as e:
            self._load_progress.visible = False
            self._loaded_info.object = f"*Error loading experiments: {e}*"
            self._analyse_btn.disabled = True
    
    def _on_analyse(self, event):
        """Run selected analysis on loaded experiments."""
        if not self._loaded_experiments:
            self._analysis_container.objects = [
                pn.pane.Alert("No experiments loaded. Go to 'Saved' tab first.", alert_type="warning")
            ]
            return
        
        analysis_type = self._analysis_type_selector.value
        
        try:
            if analysis_type == "overlay":
                self._run_overlay_analysis()
            else:
                self._run_individual_analysis()
                
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._analysis_container.objects = [
                pn.pane.Alert(f"Analysis error: {e}", alert_type="danger"),
                pn.pane.Markdown(f"```\n{tb}\n```"),
            ]
    
    def _get_conversion_params(self, exp: LoadedExperiment) -> dict:
        """Extract conversion params for an experiment (for CV calculation)."""
        flow_rate = exp.experiment_config.parameters.get("flow_rate_mL_min")
        
        try:
            column_volume_mL = calculate_column_volume_mL(
                exp.column_binding.column_parameters
            )
        except (ValueError, KeyError):
            column_volume_mL = None
        
        if flow_rate is not None and column_volume_mL is not None:
            return {
                "flow_rate_mL_min": flow_rate,
                "column_volume_mL": column_volume_mL,
            }
        return None
    
    def _run_overlay_analysis(self):
        """Create chromatogram overlay plot."""
        components = []
        
        # Header
        components.append(pn.pane.Markdown("## Chromatogram Overlay"))
        components.append(pn.pane.Markdown(f"*{len(self._loaded_experiments)} experiment(s) selected*"))
        
        # Get plot options
        normalized = self._analysis_normalized_checkbox.value
        x_axis = self._analysis_x_axis_select.value
        
        # Prepare chromatograms for overlay
        chromatograms = []
        missing_conv_params = []
        
        for exp in self._loaded_experiments:
            if exp.chromatogram_df is not None:
                label = f"{exp.experiment_set_name}/{exp.experiment_name}"
                conv_params = self._get_conversion_params(exp)
                
                if x_axis == "cv" and conv_params is None:
                    missing_conv_params.append(exp.experiment_name)
                    continue
                
                chromatograms.append((label, exp.chromatogram_df, conv_params))
        
        # Warn about missing conversion params
        if missing_conv_params:
            components.append(pn.pane.Alert(
                f"Cannot convert to CV for: {', '.join(missing_conv_params)} "
                "(missing flow rate or column dimensions)",
                alert_type="warning"
            ))
        
        if chromatograms:
            try:
                plot = plot_chromatogram_overlay_from_df(
                    chromatograms,
                    title="Chromatogram Overlay",
                    normalized=normalized,
                    x_axis=x_axis,
                    width=900,
                    height=450,
                )
                components.append(pn.pane.HoloViews(plot, sizing_mode='stretch_width'))
            except Exception as e:
                components.append(pn.pane.Alert(f"Could not create plot: {e}", alert_type="warning"))
        else:
            components.append(pn.pane.Alert("No chromatogram data available", alert_type="info"))
        
        components.append(pn.layout.Divider())
        
        # Summary table
        components.append(pn.pane.Markdown("### Selected Experiments"))
        summary_df = self._create_summary_table()
        components.append(pn.widgets.Tabulator(
            summary_df,
            height=min(200, 50 + len(self._loaded_experiments) * 30),
            sizing_mode='stretch_width',
            disabled=True,
        ))
        
        self._analysis_container.objects = components
    
    def _run_individual_analysis(self):
        """Create individual chromatogram plots."""
        components = []
        
        components.append(pn.pane.Markdown("## Individual Chromatograms"))
        components.append(pn.pane.Markdown(f"*{len(self._loaded_experiments)} experiment(s) selected*"))
        components.append(pn.layout.Divider())
        
        # Get plot options
        normalized = self._analysis_normalized_checkbox.value
        x_axis = self._analysis_x_axis_select.value
        
        for exp in self._loaded_experiments:
            components.append(pn.pane.Markdown(f"### {exp.experiment_set_name} / {exp.experiment_name}"))
            
            if exp.chromatogram_df is not None:
                conv_params = self._get_conversion_params(exp)
                
                # Check if we can do CV conversion
                if x_axis == "cv" and conv_params is None:
                    components.append(pn.pane.Alert(
                        "Cannot convert to CV (missing flow rate or column dimensions). Showing time.",
                        alert_type="warning"
                    ))
                    actual_x_axis = "minutes"
                    flow_rate = None
                    col_vol = None
                else:
                    actual_x_axis = x_axis
                    flow_rate = conv_params["flow_rate_mL_min"] if conv_params else None
                    col_vol = conv_params["column_volume_mL"] if conv_params else None
                
                try:
                    plot = plot_chromatogram_from_df(
                        exp.chromatogram_df,
                        title=exp.experiment_name,
                        normalized=normalized,
                        x_axis=actual_x_axis,
                        flow_rate_mL_min=flow_rate,
                        column_volume_mL=col_vol,
                        width=800,
                        height=350,
                    )
                    components.append(pn.pane.HoloViews(plot, sizing_mode='stretch_width'))
                except Exception as e:
                    components.append(pn.pane.Alert(f"Could not create plot: {e}", alert_type="warning"))
            else:
                components.append(pn.pane.Alert("No chromatogram data available", alert_type="info"))
            
            components.append(pn.Spacer(height=20))
        
        self._analysis_container.objects = components
    
    def _create_summary_table(self) -> pd.DataFrame:
        """Create summary table of selected experiments."""
        rows = []
        
        for exp in self._loaded_experiments:
            row = {
                "Experiment Set": exp.experiment_set_name,
                "Experiment": exp.experiment_name,
                "Column Model": exp.column_binding.column_model,
                "Binding Model": exp.column_binding.binding_model,
            }
            
            # Add some key parameters
            params = exp.experiment_config.parameters
            if "flow_rate_mL_min" in params:
                row["Flow Rate (mL/min)"] = params["flow_rate_mL_min"]
            if "gradient_start_mM" in params:
                row["Gradient Start (mM)"] = params["gradient_start_mM"]
            if "gradient_end_mM" in params:
                row["Gradient End (mM)"] = params["gradient_end_mM"]
            
            # Result info
            if exp.result.success:
                row["Status"] = "Success"
                row["Runtime (s)"] = f"{exp.result.runtime_seconds:.2f}"
            else:
                row["Status"] = "Failed"
                row["Runtime (s)"] = "-"
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _update_status(self, alert_type: str):
        """Update status pane."""
        self._status_pane.alert_type = alert_type
        self._status_pane.object = self.status
    
    def view(self) -> pn.viewable.Viewable:
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
            self._inlet_plot_area,
            sizing_mode='stretch_width',
        )
        
        # Tab 3: Simulate
        simulate_tab = pn.Column(
            pn.pane.Markdown("## 3. Run Simulations"),
            pn.Row(self._simulate_btn),
            self._simulation_progress,
            self._simulation_status_text,
            self._simulation_output,
            sizing_mode='stretch_width',
        )
        
        # Tab 4: Saved Experiments
        saved_tab = pn.Column(
            pn.pane.Markdown("## 4. Saved Experiments"),
            pn.pane.Markdown("*Select experiments to load for analysis*"),
            pn.Row(self._refresh_saved_btn, self._preview_selected_btn, self._load_data_btn),
            self._load_progress,
            self._saved_experiments_table,
            self._loaded_info,
            pn.layout.Divider(),
            self._preview_plot_area,
            sizing_mode='stretch_width',
        )
        
        # Tab 5: Analysis
        analysis_tab = pn.Column(
            pn.pane.Markdown("## 5. Analysis"),
            pn.Row(
                self._analysis_type_selector,
                self._analysis_normalized_checkbox,
                self._analysis_x_axis_select,
                self._analyse_btn,
            ),
            pn.layout.Divider(),
            self._analysis_container,
            sizing_mode='stretch_width',
        )
        
        # Assemble tabs
        tabs = pn.Tabs(
            ("1. Configure", config_tab),
            ("2. Upload", upload_tab),
            ("3. Simulate", simulate_tab),
            ("4. Saved", saved_tab),
            ("5. Analysis", analysis_tab),
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


def create_app(
    storage_dir: str = "./experiments",
    cadet_path: str | None = None,
) -> SimplifiedCADETApp:
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
    return SimplifiedCADETApp(
        storage_dir=storage_dir,
        cadet_path=cadet_path,
    )


def serve(
    storage_dir: str = "./experiments",
    cadet_path: str | None = None,
    **kwargs
):
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
    app = create_app(
        storage_dir=storage_dir,
        cadet_path=cadet_path,
    )
    pn.serve(app.view(), **kwargs)


# For panel serve
if __name__.startswith("bokeh"):
    app = create_app()
    app.view().servable()