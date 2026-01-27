"""FittingResult - results container with analysis and plotting methods.

Example
-------
    # After running fitting
    result = runner.run(problem)
    
    # Inspect results
    result.print_summary()
    print(result.fitted_parameters)
    
    # Plot comparison
    result.plot_comparison("experiment_1")
    
    # Get updated configuration
    updated_binding = result.get_updated_column_binding()
    
    # For multi-objective: plot Pareto front
    result.plot_pareto_front()
    
    # Save/export
    result.to_dict()
"""

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from copy import deepcopy

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .fitting_problem import FittingProblem
    import matplotlib.pyplot as plt


@dataclass
class FittingResult:
    """Results from a parameter fitting run.
    
    Attributes
    ----------
    problem : FittingProblem
        The problem that was solved
    success : bool
        Whether optimization converged
    fitted_parameters : dict
        Optimal parameter values
        Format: {name: value} for scalar params
        Format: {name: {component: value}} for per-component params
    final_objective : float or list[float]
        Final objective value(s)
    n_iterations : int
        Number of iterations/generations
    runtime_seconds : float
        Total runtime
    history : pd.DataFrame
        Iteration history
    pareto_x : list, optional
        Pareto front variable values (multi-objective)
    pareto_f : list, optional
        Pareto front objective values (multi-objective)
    """
    
    problem: "FittingProblem"
    success: bool
    fitted_parameters: dict[str, Any]
    final_objective: float | list[float]
    n_iterations: int
    runtime_seconds: float
    history: pd.DataFrame
    pareto_x: list | None = None
    pareto_f: list | None = None
    
    # Internal references
    _opt_results: Any = field(default=None, repr=False)
    _opt_problem: Any = field(default=None, repr=False)
    
    def print_summary(self):
        """Print a summary of the fitting results."""
        print(f"Fitting: {self.problem.name}")
        print(f"Status: {'✓ Success' if self.success else '✗ Failed'}")
        print(f"Iterations: {self.n_iterations}")
        print(f"Runtime: {self.runtime_seconds:.1f}s")
        
        # Format objective
        if isinstance(self.final_objective, (list, np.ndarray)):
            obj_str = ", ".join(f"{v:.4g}" for v in self.final_objective)
            print(f"Final objective: [{obj_str}]")
        else:
            print(f"Final objective: {self.final_objective:.4g}")
        
        print("\nFitted parameters:")
        for name, value in self.fitted_parameters.items():
            if isinstance(value, dict):
                # Per-component parameter
                print(f"  {name}:")
                for comp, v in value.items():
                    print(f"    {comp}: {v:.6g}")
            else:
                print(f"  {name}: {value:.6g}")
        
        # Multi-objective info
        if self.pareto_x is not None:
            print(f"\nPareto front: {len(self.pareto_x)} solutions")
    
    def get_updated_column_binding(
        self,
        experiment_name: str | None = None,
    ) -> Any:
        """Get ColumnBindingConfig with fitted parameters applied.
        
        Parameters
        ----------
        experiment_name : str, optional
            For experiment-specific variables, which experiment's values to use.
            If None, uses shared variable values only (experiment-specific 
            variables are skipped with a warning).
        
        Returns
        -------
        ColumnBindingConfig
            Configuration with fitted values
            
        Notes
        -----
        Since CADET parameter names are used directly (adsorption_rate, 
        desorption_rate, etc.), the mapping to ColumnBindingConfig is 
        straightforward.
        """
        import warnings
        
        updated = deepcopy(self.problem.column_binding)
        
        # Track which parameters were applied
        applied = []
        skipped = []
        
        for var in self.problem.selected_variables:
            base_name = var.parameter.name
            
            # Determine the key to look up in fitted_parameters
            if var.is_experiment_specific:
                if experiment_name is None:
                    # Skip experiment-specific variables
                    skipped.append(base_name)
                    continue
                key = f"{base_name}_{experiment_name}"
            else:
                key = base_name
            
            if key not in self.fitted_parameters:
                continue
            
            value = self.fitted_parameters[key]
            applied.append(base_name)
            
            if var.parameter.per_component:
                # Update component parameters
                # Determine target based on category
                is_binding = var.parameter.category.value.startswith("binding")
                
                if is_binding:
                    target = getattr(updated, 'component_binding_parameters', {})
                else:
                    target = getattr(updated, 'component_column_parameters', {})
                
                # Initialize array if needed
                if base_name not in target:
                    n_comp = len(self.problem._component_names)
                    # Try to get current values, else initialize to zeros
                    try:
                        current = getattr(
                            updated, base_name, [0.0] * n_comp
                        )
                        if hasattr(current, 'tolist'):
                            current = current.tolist()
                        target[base_name] = list(current)
                    except Exception:
                        target[base_name] = [0.0] * n_comp
                
                # Update fitted components
                for comp_name, v in value.items():
                    idx = self.problem._component_names.index(comp_name)
                    target[base_name][idx] = v
                
                # Set back on object
                if is_binding:
                    updated.component_binding_parameters = target
                else:
                    updated.component_column_parameters = target
            else:
                # Update scalar parameter
                is_binding = var.parameter.category.value.startswith("binding")
                
                if is_binding:
                    params = getattr(updated, 'binding_parameters', {})
                    params[base_name] = value
                    updated.binding_parameters = params
                else:
                    params = getattr(updated, 'column_parameters', {})
                    params[base_name] = value
                    updated.column_parameters = params
        
        # Warn about skipped experiment-specific variables
        if skipped and experiment_name is None:
            warnings.warn(
                f"Experiment-specific variables were skipped (no experiment_name "
                f"specified): {skipped}. Call with experiment_name to include them."
            )
        
        return updated
    
    def simulate_with_fitted(
        self,
        experiment_name: str | None = None,
    ) -> list:
        """Run simulations with fitted parameters.
        
        Parameters
        ----------
        experiment_name : str, optional
            Specific experiment to simulate. If None, simulates all.
            
        Returns
        -------
        list
            Simulation results
        """
        # Import here to avoid circular imports
        from CADETProcess.simulator import Cadet
        
        updated_binding = self.get_updated_column_binding()
        
        experiments = self.problem.experiments
        if experiment_name:
            experiments = [e for e in experiments if e.name == experiment_name]
        
        results = []
        simulator = Cadet()
        
        for exp in experiments:
            process = self.problem.operation_mode.create_process(
                exp, updated_binding
            )
            sim_result = simulator.simulate(process)
            results.append({
                'experiment': exp.name,
                'process': process,
                'simulation': sim_result,
            })
        
        return results
    
    def plot_comparison(
        self,
        experiment_name: str,
        components: list[str] | None = None,
        ax: "plt.Axes | None" = None,
        time_units: str = "min",
    ):
        """Plot reference vs fitted simulation for one experiment.
        
        Parameters
        ----------
        experiment_name : str
            Experiment to plot
        components : list[str], optional
            Components to plot (default: all in reference)
        ax : matplotlib.Axes, optional
            Axes to plot on
        time_units : str
            Time units: "min" or "s"
            
        Returns
        -------
        matplotlib.Axes
            The axes object
        """
        import matplotlib.pyplot as plt
        
        # Get reference data
        if experiment_name not in self.problem.reference_data:
            raise ValueError(
                f"No reference data for experiment '{experiment_name}'"
            )
        ref = self.problem.reference_data[experiment_name]
        
        # Run simulation with fitted parameters
        sim_results = self.simulate_with_fitted(experiment_name)
        if not sim_results:
            raise ValueError(f"Could not simulate experiment '{experiment_name}'")
        
        sim_data = sim_results[0]
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine components to plot
        if components is None:
            components = ref.component_names
        
        # Time conversion
        time_factor = 60.0 if time_units == "min" else 1.0
        time_label = "Time (min)" if time_units == "min" else "Time (s)"
        
        time_ref = ref.time / time_factor
        
        # Color palette
        colors = plt.cm.tab10.colors
        
        for i, comp in enumerate(components):
            color = colors[i % len(colors)]
            
            # Plot reference (markers)
            if comp in ref.concentrations:
                ax.plot(
                    time_ref,
                    ref.concentrations[comp],
                    'o',
                    color=color,
                    label=f'{comp} (reference)',
                    markersize=4,
                    alpha=0.7,
                )
            
            # Plot simulation (line)
            sim = sim_data['simulation']
            if hasattr(sim, 'solution'):
                # Extract solution for this component
                try:
                    time_sim = sim.solution.outlet.outlet.time / time_factor
                    
                    # Get component index
                    comp_idx = self.problem._component_names.index(comp)
                    conc_sim = sim.solution.outlet.outlet.solution[:, comp_idx]
                    
                    ax.plot(
                        time_sim,
                        conc_sim,
                        '-',
                        color=color,
                        label=f'{comp} (fitted)',
                        linewidth=1.5,
                    )
                except Exception as e:
                    # Try alternative solution access
                    pass
        
        ax.set_xlabel(time_label)
        ax.set_ylabel('Concentration (mM)')
        ax.set_title(f'Fitting Result: {experiment_name}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return ax
    
    def plot_pareto_front(
        self,
        ax: "plt.Axes | None" = None,
        highlight_index: int | None = None,
    ):
        """Plot Pareto front for multi-objective results.
        
        Parameters
        ----------
        ax : matplotlib.Axes, optional
            Axes to plot on
        highlight_index : int, optional
            Index of solution to highlight
            
        Returns
        -------
        matplotlib.Axes
            The axes object
        """
        if self.pareto_f is None:
            raise ValueError(
                "No Pareto front available (single-objective optimization)"
            )
        
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        pareto_f = np.array(self.pareto_f)
        
        if pareto_f.shape[1] == 2:
            # 2D Pareto front
            ax.scatter(
                pareto_f[:, 0],
                pareto_f[:, 1],
                c='blue',
                s=50,
                alpha=0.7,
                label='Pareto front',
            )
            
            if highlight_index is not None:
                ax.scatter(
                    pareto_f[highlight_index, 0],
                    pareto_f[highlight_index, 1],
                    c='red',
                    s=100,
                    marker='*',
                    label='Selected',
                    zorder=5,
                )
            
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
        else:
            # Higher dimensional: use parallel coordinates
            df = pd.DataFrame(
                pareto_f,
                columns=[f'Obj_{i+1}' for i in range(pareto_f.shape[1])]
            )
            pd.plotting.parallel_coordinates(
                df.assign(idx=range(len(df))),
                'idx',
                ax=ax,
                colormap='viridis',
            )
            ax.get_legend().remove()
        
        ax.set_title('Pareto Front')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def plot_convergence(
        self,
        ax: "plt.Axes | None" = None,
    ):
        """Plot optimization convergence history.
        
        Parameters
        ----------
        ax : matplotlib.Axes, optional
            Axes to plot on
            
        Returns
        -------
        matplotlib.Axes
            The axes object
        """
        import matplotlib.pyplot as plt
        
        if self.history.empty:
            raise ValueError("No history available")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Try to extract generation/iteration and objective info
        if 'generation' in self.history.columns and 'f' in self.history.columns:
            # Group by generation, get best objective
            best_per_gen = []
            for gen, group in self.history.groupby('generation'):
                f_values = group['f'].apply(
                    lambda x: sum(x) if isinstance(x, (list, np.ndarray)) else x
                )
                best_per_gen.append({
                    'generation': gen,
                    'best_f': f_values.min(),
                })
            
            df = pd.DataFrame(best_per_gen)
            ax.plot(df['generation'], df['best_f'], 'b-', linewidth=2)
            ax.set_xlabel('Generation')
        else:
            # Simple iteration plot if available
            ax.plot(range(len(self.history)), self.history.iloc[:, 0], 'b-')
            ax.set_xlabel('Iteration')
        
        ax.set_ylabel('Best Objective')
        ax.set_title('Optimization Convergence')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def select_pareto_solution(self, index: int) -> "FittingResult":
        """Select a specific solution from the Pareto front.
        
        Parameters
        ----------
        index : int
            Index of solution in Pareto front
            
        Returns
        -------
        FittingResult
            New result with selected solution
        """
        if self.pareto_x is None:
            raise ValueError("No Pareto front available")
        
        if index < 0 or index >= len(self.pareto_x):
            raise ValueError(
                f"Index {index} out of range [0, {len(self.pareto_x)})"
            )
        
        # Map x values to parameters
        from .fitting_runner import FittingRunner
        runner = FittingRunner()
        new_params = runner._map_x_to_parameters(
            self.problem, 
            self.pareto_x[index]
        )
        
        return FittingResult(
            problem=self.problem,
            success=self.success,
            fitted_parameters=new_params,
            final_objective=self.pareto_f[index],
            n_iterations=self.n_iterations,
            runtime_seconds=self.runtime_seconds,
            history=self.history,
            pareto_x=self.pareto_x,
            pareto_f=self.pareto_f,
            _opt_results=self._opt_results,
            _opt_problem=self._opt_problem,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.
        
        Returns
        -------
        dict
            Serializable dictionary
        """
        return {
            "name": self.problem.name,
            "success": self.success,
            "fitted_parameters": self._serialize_parameters(),
            "final_objective": (
                self.final_objective 
                if isinstance(self.final_objective, (int, float))
                else list(self.final_objective)
            ),
            "n_iterations": self.n_iterations,
            "runtime_seconds": self.runtime_seconds,
            "experiments_fitted": self.problem.experiments_to_fit,
            "variables_fitted": [
                v.parameter.name for v in self.problem.selected_variables
            ],
            "metric": self.problem.metric,
            "has_pareto_front": self.pareto_x is not None,
            "pareto_size": len(self.pareto_x) if self.pareto_x else 0,
        }
    
    def _serialize_parameters(self) -> dict:
        """Serialize fitted parameters to JSON-safe format."""
        result = {}
        for name, value in self.fitted_parameters.items():
            if isinstance(value, dict):
                result[name] = {k: float(v) for k, v in value.items()}
            elif isinstance(value, np.ndarray):
                result[name] = value.tolist()
            else:
                result[name] = float(value)
        return result
    
    def __repr__(self) -> str:
        status = "success" if self.success else "failed"
        return (
            f"FittingResult({self.problem.name}, {status}, "
            f"obj={self.final_objective:.4g if isinstance(self.final_objective, (int, float)) else 'multi'})"
        )
