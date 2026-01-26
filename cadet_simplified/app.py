"""Simplified Panel GUI for CADET chromatography simulations.

A streamlined interface with Excel-based configuration:
1. Select models and components
2. Download template
3. Upload filled template
4. Validate and simulate
5. Browse saved experiments
6. Analyse selected experiments
"""

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING

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
from .storage import FileResultsStorage, LoadedExperiment
from .simulation import SimulationRunner, ValidationResult
from .analysis import AnalysisView, get_analysis, list_analyses
from .utils.path_utils import get_storage_path
if TYPE_CHECKING:
    from .operation_modes import ExperimentConfig, ColumnBindingConfig


def _detect_cadet_path() -> tuple[str | None, str]:
    """Try to detect CADET installation path.
    
    Detection order:
    1. CADET-Process auto-detection
    2. CADET_PATH environment variable
    3. None (with error message)
    
    Returns
    -------
    tuple[str | None, str]
        (detected_path, status_message)
        - detected_path: Path string if found, None if not
        - status_message: Description of detection result
    """
    # 1. Try CADET-Process auto-detection
    try:
        from CADETProcess.simulator import Cadet
        simulator = Cadet()
        simulator.check_cadet()
        
        if simulator.cadet_path:
            cadet_path = Path(simulator.cadet_path)
            if cadet_path.is_file():
                path = str(cadet_path.parent)
            elif cadet_path.is_dir():
                path = str(cadet_path)
            else:
                path = str(simulator.cadet_path)
            return path, "Auto-detected via CADET-Process"
    except Exception:
        pass
    
    # 2. Try environment variable
    env_path = os.environ.get("CADET_PATH")
    if env_path:
        if Path(env_path).exists():
            return env_path, "Loaded from CADET_PATH environment variable"
        else:
            # Environment variable set but path doesn't exist
            return None, f"CADET_PATH environment variable set to '{env_path}' but path does not exist"
    
    # 3. Detection failed
    return None, "CADET path not found"


CADET_PATH_HELP = """
### CADET Path Not Found

CADET could not be auto-detected. Please either:

**Option 1: Set environment variable (recommended)**
```bash
# Linux/Mac - add to ~/.bashrc or ~/.zshrc
export CADET_PATH="/path/to/cadet/bin"

# Windows - in Command Prompt
set CADET_PATH=C:\\path\\to\\cadet\\bin
```

**Option 2: Enter the path manually below**

The path should point to the directory containing the CADET executable 
(e.g., `/home/user/miniconda3/envs/cadet/bin` or `C:\\cadet\\bin`).
"""


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
        storage_dir: str | Path = None,
        cadet_path: str | None = None,
        n_load_workers: int = 4,
        **params
    ):
        super().__init__(**params)
        if not storage_dir:
            storage_dir = get_storage_path()
        self.storage_dir = Path(storage_dir)
        self.cadet_path = cadet_path
        self.n_load_workers = n_load_workers
        
        # Initialize storage and runner
        self.storage = FileResultsStorage(self.storage_dir)
        self.runner = SimulationRunner(cadet_path)
        
        # State
        self._current_parse_result: ParseResult | None = None
        self._simulation_results: list = []
        self._component_names: list[str] = self._default_component_names()
        
        # Analysis state
        self._loaded_experiments: list[LoadedExperiment] = []
        self._analysis_view = AnalysisView()
        
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
        # CADET path detection
        if self.cadet_path:
            # Path explicitly provided
            detected_path = self.cadet_path
            path_status = "Provided via configuration"
            path_found = True
        else:
            # Try auto-detection
            detected_path, path_status = _detect_cadet_path()
            path_found = detected_path is not None
        
        self._cadet_path_input = pn.widgets.TextInput(
            name="CADET Path",
            value=detected_path or "",
            placeholder="/path/to/cadet/bin",
            width=500,
        )
        
        self._cadet_path_status = pn.pane.Markdown(
            f"*{path_status}*" if path_found else "",
            sizing_mode='stretch_width',
        )
        
        self._cadet_path_help = pn.pane.Markdown(
            CADET_PATH_HELP if not path_found else "",
            sizing_mode='stretch_width',
        )
        
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
        
        # Quick results plot (for immediate feedback after simulation)
        self._quick_results_plot = pn.pane.HoloViews(
            None,
            sizing_mode='stretch_width',
            height=400,
        )
        
        # === Tab 4: Saved Experiments ===
        self._saved_experiments_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            height=350,
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
        analysis_options = {a["description"]: a["name"] for a in list_analyses()}
        self._analysis_type_selector = pn.widgets.Select(
            name="Analysis Type",
            options=analysis_options,
            value=list(analysis_options.values())[0] if analysis_options else None,
            width=300,
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
            file_bytes = io.BytesIO(event.new)
            parser = ExcelParser()
            result = parser.parse(file_bytes)
            
            self._current_parse_result = result
            
            if not result.success:
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
            
            n_exp = len(result.experiments)
            self.status = f"Parsed {n_exp} experiment(s). Click 'Validate Configuration' to check."
            self._update_status("info")
            
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
        
        # Validate CADET path
        cadet_path = self._cadet_path_input.value.strip()
        if not cadet_path:
            self.status = "Error: CADET path is required for validation. Please enter the path in the Simulate tab."
            self._update_status("danger")
            return
        
        if not Path(cadet_path).exists():
            self.status = f"Error: CADET path does not exist: {cadet_path}"
            self._update_status("danger")
            return
        
        # Update runner with current CADET path
        self.runner = SimulationRunner(cadet_path)
        
        result = self._current_parse_result
        mode = get_operation_mode(self.operation_mode)
        
        validation_results = []
        all_valid = True
        
        for exp in result.experiments:
            try:
                process = mode.create_process(exp, result.column_binding)
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
        """Run simulations and save results to storage."""
        if self._current_parse_result is None:
            return
        
        # Validate CADET path
        cadet_path = self._cadet_path_input.value.strip()
        if not cadet_path:
            self.status = "Error: CADET path is required. Please enter the path to your CADET installation."
            self._update_status("danger")
            return
        
        if not Path(cadet_path).exists():
            self.status = f"Error: CADET path does not exist: {cadet_path}"
            self._update_status("danger")
            return
        
        # Update runner with current CADET path
        self.runner = SimulationRunner(cadet_path)
        
        result = self._current_parse_result
        mode = get_operation_mode(self.operation_mode)
        
        self._simulation_results = []
        self._simulation_progress.visible = True
        self._simulation_progress.value = 0
        
        n_experiments = len(result.experiments)
        output_items = [pn.pane.Markdown("### Simulation Progress\n")]
        
        # Use storage's pending directory for H5 files
        h5_dir = self.storage.get_pending_dir()
        
        for i, exp in enumerate(result.experiments):
            progress = int((i / n_experiments) * 100)
            self._simulation_progress.value = progress
            self.status = f"Simulating {exp.name} ({i+1}/{n_experiments})..."
            self._update_status("info")
            
            try:
                process = mode.create_process(exp, result.column_binding)
                sim_result = self.runner.run(process, h5_dir=h5_dir)
                self._simulation_results.append(sim_result)
                
                if sim_result.success:
                    output_items.append(
                        pn.pane.Markdown(f"✓ **{exp.name}**: Completed in {sim_result.runtime_seconds:.2f}s")
                    )
                else:
                    error_msg = "; ".join(sim_result.errors[:2])
                    output_items.append(
                        pn.pane.Markdown(f"❌ **{exp.name}**: Failed - {error_msg}")
                    )
                    
            except Exception as e:
                from .simulation import SimulationResultWrapper
                self._simulation_results.append(SimulationResultWrapper(
                    experiment_name=exp.name,
                    success=False,
                    errors=[str(e)],
                ))
                output_items.append(
                    pn.pane.Markdown(f"❌ **{exp.name}**: Error - {str(e)}")
                )
        
        self._simulation_progress.value = 100
        self._simulation_progress.visible = False
        
        successful = sum(1 for r in self._simulation_results if r.success)
        self.status = f"Completed: {successful}/{n_experiments} simulations successful."
        self._update_status("success" if successful == n_experiments else "warning")
        
        self._simulation_output.objects = output_items
        
        # Save to storage if any successful
        if successful > 0:
            try:
                # Generate a name based on timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                set_name = f"Simulation_{timestamp}"
                
                set_id = self.storage.save_experiment_set(
                    name=set_name,
                    operation_mode=self.operation_mode,
                    experiments=result.experiments,
                    column_binding=result.column_binding,
                    results=self._simulation_results,
                )
                
                output_items.append(pn.pane.Markdown(f"\n✓ Results saved. Set ID: `{set_id}`"))
                output_items.append(pn.pane.Markdown(f"✓ Excel export saved: `{set_name}.xlsx`"))
                output_items.append(pn.pane.Markdown("*Go to the 'Saved' tab to browse and analyze results.*"))
                self._simulation_output.objects = output_items
                
            except Exception as e:
                output_items.append(pn.pane.Markdown(f"\n⚠️ Warning: Could not save results: {e}"))
                self._simulation_output.objects = output_items
    
    def _on_refresh_saved(self, event):
        """Refresh saved experiments table."""
        try:
            df = self.storage.list_experiments(limit=25)
            
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
                
        except Exception as e:
            self._loaded_info.object = f"*Error loading experiments: {e}*"
    
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
        
        # Build selection list: [(set_id, exp_name), ...]
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
                n_workers=self.n_load_workers,
            )
            
            self._load_progress.value = 100
            self._load_progress.visible = False
            
            n_loaded = len(self._loaded_experiments)
            self._loaded_info.object = f"✓ **{n_loaded}** experiment(s) loaded. Go to 'Analysis' tab to analyze."
            
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
        
        analysis_name = self._analysis_type_selector.value
        if not analysis_name:
            return
        
        try:
            # Clear and run analysis
            self._analysis_view.clear()
            analysis = get_analysis(analysis_name)
            analysis.run(self._loaded_experiments, self._analysis_view)
            
            # Update container
            self._analysis_container.objects = [self._analysis_view.view()]
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._analysis_container.objects = [
                pn.pane.Alert(f"Analysis error: {e}", alert_type="danger"),
                pn.pane.Markdown(f"```\n{tb}\n```"),
            ]
    
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
            sizing_mode='stretch_width',
        )
        
        # Tab 3: Simulate
        simulate_tab = pn.Column(
            pn.pane.Markdown("## 3. Run Simulations"),
            pn.pane.Markdown("### CADET Configuration"),
            self._cadet_path_input,
            self._cadet_path_status,
            self._cadet_path_help,
            pn.layout.Divider(),
            pn.Row(self._simulate_btn),
            self._simulation_progress,
            self._simulation_output,
            sizing_mode='stretch_width',
        )
        
        # Tab 4: Saved Experiments (was Tab 5)
        saved_tab = pn.Column(
            pn.pane.Markdown("## 4. Saved Experiments"),
            pn.pane.Markdown("*Select experiments to load for analysis*"),
            pn.Row(self._refresh_saved_btn, self._load_data_btn),
            self._load_progress,
            self._saved_experiments_table,
            self._loaded_info,
            sizing_mode='stretch_width',
        )
        
        # Tab 5: Analysis (new)
        analysis_tab = pn.Column(
            pn.pane.Markdown("## 5. Analysis"),
            pn.Row(
                self._analysis_type_selector,
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
    n_load_workers: int = 4,
) -> SimplifiedCADETApp:
    """Create the application.
    
    Parameters
    ----------
    storage_dir : str
        Directory for storing experiment data
    cadet_path : str, optional
        Path to CADET installation
    n_load_workers : int, default=4
        Number of workers for parallel loading
        
    Returns
    -------
    SimplifiedCADETApp
        The application instance
    """
    return SimplifiedCADETApp(
        storage_dir=storage_dir,
        cadet_path=cadet_path,
        n_load_workers=n_load_workers,
    )


def serve(
    storage_dir: str = "./experiments",
    cadet_path: str | None = None,
    n_load_workers: int = 4,
    **kwargs
):
    """Serve the application.
    
    Parameters
    ----------
    storage_dir : str
        Directory for storing experiment data
    cadet_path : str, optional
        Path to CADET installation
    n_load_workers : int, default=4
        Number of workers for parallel loading
    **kwargs
        Additional arguments passed to pn.serve()
    """
    app = create_app(
        storage_dir=storage_dir,
        cadet_path=cadet_path,
        n_load_workers=n_load_workers,
    )
    pn.serve(app.view(), **kwargs)


# For running directly
if __name__ == "__main__":
    serve(port=5007, show=True)
