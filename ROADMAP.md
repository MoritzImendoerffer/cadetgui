# CADET-GUI Roadmap

## Vision

A Jupyter-based GUI for CADET-Process that enables:
1. **Forward simulation** — Configure and run chromatography simulations without code
2. **Parameter estimation** — Fit model parameters to experimental data
3. **Config management** — Save/load configurations as JSON files

---

## Current State (v0.3)

### ✅ Implemented

**Core Modules:**
| Module | Status | Description |
|--------|--------|-------------|
| `config.py` | ✅ Complete | Dataclasses for JSON config, supports multiple experiments |
| `units.py` | ✅ Complete | Unit conversions (cm→m, mM→mol/m³, CV→seconds) |
| `builder.py` | ✅ Complete | Builds CADET-Process objects, supports per-experiment Process |
| `estimation.py` | ✅ Complete | Multi-experiment estimation with shared parameters |
| `io.py` | ✅ Complete | CSV/Excel template generation |

**GUI Tabs:**
| Tab | Status | What Works |
|-----|--------|------------|
| Process Setup | ✅ Complete | Components, column model, dynamic binding params, simulate, config I/O |
| Experimental Data | ✅ Complete | Multiple experiments, add/remove, per-experiment conditions and data |
| Parameter Estimation | ✅ Complete | Variable selection, bounds, multi-experiment optimization |

**JSON Config:**
- ✅ Load from JSON (requires `experiments` list format)
- ✅ Save to JSON (uses `experiments` list format)
- ✅ Load into GUI via file upload
- ✅ Export from GUI via download button

---

## Phase 1: Forward Simulation MVP ✅

**Goal:** User can configure a gradient elution experiment and see the simulated chromatogram.

### 1.1 Component Editor ✅
- [x] Add component table (2 rows: Salt, Product)
- [x] Editable: name, molecular weight

### 1.2 Column Model Selector ✅
- [x] Dropdown: LumpedRateModelWithPores (default), LumpedRateModelWithoutPores, GeneralRateModel
- [x] Update builder.py to use selected model
- [ ] (Future: GRM-specific params like pore diffusion)

### 1.3 Dynamic Binding Model Parameters ✅
- [x] Show/hide parameters based on binding model type:
  - **SMA:** lambda_, ka, kd, nu, sigma (per non-salt component)
  - **Langmuir:** ka, kd (per component)
  - **Linear:** ka (per component)

### 1.4 Simulate Button + Chromatogram Plot ✅
- [x] "Simulate" button in Process Setup tab
- [x] Calls `builder.build_process()` → `Cadet().simulate()`
- [x] Plots outlet concentration vs time
- [x] Error handling if CADET not installed
- [ ] Loading indicator during simulation (optional enhancement)

### 1.5 Flow Sheet Display (Info Only) ✅
- [x] Show text: "inlet → column → outlet"
- [ ] Display computed values: cycle time, gradient duration (optional)
- [ ] (Future: visual diagram)

---

## Phase 2: Config Template Interface

**Goal:** User can download/upload complete JSON configs.

### 2.1 Config Download
- [x] "Download Config" button in Process Setup
- [x] Exports current GUI state as JSON file

### 2.2 Config Upload
- [x] "Load Config" file input (accepts .json)
- [x] Populates all GUI fields from uploaded config
- [x] Validates config structure, shows errors

### 2.3 Data Template
- [x] Template download button in Data tab (already wired)
- [ ] Template includes example data matching config timeframe (optional enhancement)

---

## Phase 3: Parameter Estimation Integration ✅

**Goal:** User can fit binding parameters to experimental data from multiple experiments.

### 3.1 Multi-Experiment Support ✅
- [x] Data tab supports multiple experiments
- [x] Add/remove experiments with different gradient conditions
- [x] Each experiment has own data file
- [x] Shared column and binding model across experiments

### 3.2 Prerequisites Check ✅
- [x] Verify at least one experiment has data loaded
- [x] Verify at least one parameter selected
- [x] Verify CADET-Process installed

### 3.3 Run Estimation ✅
- [x] Build Process for each experiment
- [x] Create Comparator for each with reference data
- [x] Shared variables (binding parameters) across all processes
- [x] Sum of objectives (SSE) across experiments
- [ ] Progress indicator (optional enhancement)
- [ ] Cancel button (optional)

### 3.4 Results Display ✅
- [x] Fitted parameter values
- [x] Objective value
- [x] Success/failure status
- [ ] Overlay plot: simulation vs experimental (future)
- [ ] Residuals plot (future)

---

## Phase 4: Future Enhancements

### Results Visualization
- [ ] Overlay plot: simulation vs experimental data
- [ ] Residuals plot
- [ ] Parameter correlation plots
- [ ] Export results to CSV/Excel

### Advanced Column Models
- [ ] GeneralRateModel with pore diffusion parameters
- [ ] 2D-GRM for radial effects
- [ ] Custom film/pore diffusion correlations

### Flow Sheet Editor
- [ ] Visual drag-and-drop flow sheet builder
- [ ] Support for valves, multiple inlets/outlets
- [ ] Breakthrough experiments (step input)
- [ ] Load-wash-elute sequences

### Advanced Optimization
- [ ] Multi-objective (U-NSGA3)
- [ ] Global optimizers (Ax, BOBYQA)
- [ ] Sensitivity analysis
- [ ] Confidence intervals

### Integration
- [ ] CADET-Workshop example loader
- [ ] Export to CADET-Process Python script
- [ ] Batch processing multiple configs

---

## Technical Notes

### Unit Convention
All user-facing values use practical units:
- Length: cm, mm, μm
- Flow: mL/min
- Concentration: mM, g/L
- Time: seconds (or CV for gradient)

Conversion to SI happens in `builder.py` when creating CADET objects.

### Config JSON Structure
```json
{
  "name": "Experiment Name",
  "components": [
    {"name": "Salt", "molecular_weight": 58.44},
    {"name": "Product", "molecular_weight": 150000}
  ],
  "column": {
    "type": "LumpedRateModelWithPores",
    "length_cm": 10.0,
    "diameter_mm": 7.7,
    "bed_porosity": 0.37,
    "particle_radius_um": 34.0,
    "particle_porosity": 0.33,
    "axial_dispersion": 1e-7,
    "film_diffusion": [1e-4, 1e-6]
  },
  "binding_model": {
    "type": "StericMassAction",
    "is_kinetic": true,
    "lambda_": 800,
    "adsorption_rate": [0, 35.5],
    "desorption_rate": [0, 1000],
    "characteristic_charge": [0, 4.7],
    "steric_factor": [0, 10]
  },
  "experiments": [
    {
      "name": "5CV Gradient",
      "data_file": "data_5cv.csv",
      "gradient_start_mM": 50,
      "gradient_end_mM": 500,
      "gradient_cv": 5,
      "flow_rate_mL_min": 1.0,
      "load_concentration_g_L": 1.0
    },
    {
      "name": "30CV Gradient",
      "data_file": "data_30cv.csv",
      "gradient_start_mM": 50,
      "gradient_end_mM": 500,
      "gradient_cv": 30,
      "flow_rate_mL_min": 1.0,
      "load_concentration_g_L": 1.0
    }
  ],
  "variables": [
    {
      "path": "flow_sheet.column.binding_model.adsorption_rate",
      "component": 1,
      "bounds": [0.1, 100],
      "transform": "log"
    }
  ]
}
```

### Dependencies
- **Required:** panel, param, pandas, numpy
- **Optional:** matplotlib (plotting), cadet-process (simulation)
- **For estimation:** scipy, cadet-process

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-01 | Initial structure, basic GUI tabs |
| 0.1.1 | 2025-01 | Fixed unit mismatch (cm/mm/μm vs m) |
| 0.2.0 | 2025-01 | Phase 1 complete: components, column model, dynamic binding, simulate button |
| 0.2.1 | 2025-01 | Phase 2 complete: config download/upload |
| 0.3.0 | 2025-01 | Phase 3 complete: multi-experiment support, parameter estimation |
| 0.4.0 | TBD | Phase 4: Results visualization, advanced features |
