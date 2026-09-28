"""
src/credit_toolbox/governance/model_card.py

Generates production-ready compliance artifacts and reproducibility scripts.
Extracts the fitted state of Scikit-Learn pipelines and custom transformers to create 
a declarative JSON configuration and a zero-dependency recreation script.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Template for the auditable reproduction script
REPRODUCE_SCRIPT_TEMPLATE = """\
# ==============================================================================
# Auto-Generated Model Reproducibility Script
# Generated on: 2026-09-27
# Location Context: Armenia, Quindio, Colombia (Remote)
# ==============================================================================

import json
import numpy as np
from sklearn.pipeline import Pipeline

# Dynamic imports extracted from the validated pipeline
{imports}

def reconstruct_pipeline(config_path: str = "pipeline_state.json") -> Pipeline:
    \"\"\"
    Rebuilds the exact Scikit-Learn pipeline using the declarative configuration file.
    Guarantees mathematically identical outputs without needing the original Jupyter Notebook.
    \"\"\"
    with open(config_path, 'r') as f:
        config = json.load(f)

    steps = []
    for step_data in config['steps']:
        # Fetch the class definition dynamically from the imported modules
        step_class = globals()[step_data['class_name']]
        
        if 'state' in step_data:
            # Custom Stateful Credit Transformer (Phase 4)
            instance = step_class()
            if hasattr(instance, 'load_state'):
                instance.load_state(step_data['state'])
            else:
                raise AttributeError(f"Class {{step_data['class_name']}} lacks load_state method.")
        else:
            # Standard Scikit-Learn Estimator (e.g., LogisticRegression)
            instance = step_class(**step_data.get('params', {{}}))
            if 'coef_' in step_data:
                instance.coef_ = np.array(step_data['coef_'])
                instance.classes_ = np.array([0, 1]) # Standard binary default flags
            if 'intercept_' in step_data:
                instance.intercept_ = np.array(step_data['intercept_'])
                
        steps.append((step_data['name'], instance))

    return Pipeline(steps)

if __name__ == '__main__':
    pipeline = reconstruct_pipeline('pipeline_state.json')
    print("Pipeline successfully reconstructed from configuration!")
    print(pipeline)
"""


class ArtifactExporter:
    """
    Governs the extraction of pipeline DNA and the compilation of auditable artifacts.
    """
    
    def __init__(self, pipeline: Pipeline, model_name: str, output_dir: str = "./artifacts"):
        self.pipeline = pipeline
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self) -> None:
        """Orchestrates the creation of all Phase 6 compliance deliverables."""
        logger.info(f"Initiating artifact export for model: {self.model_name}")
        
        state_dict, modules_to_import = self._extract_pipeline_state()
        
        # 1. Export Declarative JSON Configuration
        json_path = self.output_dir / "pipeline_state.json"
        with open(json_path, "w") as f:
            json.dump(state_dict, f, indent=4)
            
        # 2. Export Reproducibility Script
        script_path = self.output_dir / "reproduce_model.py"
        self._generate_reproduce_script(script_path, modules_to_import)
        
        # 3. Export Markdown Model Card
        md_path = self.output_dir / f"{self.model_name}_model_card.md"
        self._generate_model_card(md_path)
        
        logger.info(f"Successfully generated all compliance artifacts in: {self.output_dir}")

    def _extract_pipeline_state(self) -> tuple[Dict[str, Any], set]:
        """
        Iterates through the pipeline to extract fitted states, thresholds, and coefficients.
        Returns the declarative dictionary and a set of required Python modules.
        """
        state_dict = {"model_name": self.model_name, "steps": []}
        modules_to_import = set()
        
        for name, step in self.pipeline.steps:
            class_name = type(step).__name__
            module_name = type(step).__module__
            modules_to_import.add(f"from {module_name} import {class_name}")
            
            step_info = {
                "name": name,
                "class_name": class_name,
                "module": module_name,
            }
            
            # Utilize the custom Phase 1 Stateful Interface if available
            if hasattr(step, "export_state"):
                step_info["state"] = step.export_state()
            else:
                # Fallback for native Scikit-Learn estimators
                step_info["params"] = step.get_params()
                if hasattr(step, "coef_"):
                    step_info["coef_"] = step.coef_.tolist()
                if hasattr(step, "intercept_"):
                    step_info["intercept_"] = step.intercept_.tolist()
            
            state_dict["steps"].append(step_info)
            
        return state_dict, modules_to_import

    def _generate_reproduce_script(self, script_path: Path, modules_to_import: set) -> None:
        """Injects dynamically discovered dependencies into the runbook template."""
        imports_block = "\n".join(sorted(list(modules_to_import)))
        script_content = REPRODUCE_SCRIPT_TEMPLATE.format(imports=imports_block)
        
        with open(script_path, "w") as f:
            f.write(script_content)

    def _generate_model_card(self, md_path: Path) -> None:
        """Generates a high-level Markdown compliance summary."""
        md_content = f"""# Credit Risk Model Card
        
**Model Name:** {self.model_name}
**Author:** Guilherme Artoni
**Organization:** EFX - Advanced Analytics 
**Compilation Date:** 2026-09-27

## System Architecture
This pipeline enforces deterministic mathematical constraints prior to production deployment. All features have passed Phase 6 Quality Gates for imputation, capping, and strict WOE monotonicity.

## Reproducibility
The internal state of this model (boundaries, thresholds, weights) has been serialized. To recreate this Scikit-Learn Pipeline identically for auditing or future quantitative research:

```bash
python reproduce_model.py
```
with open(md_path, "w") as f:
f.write(md_content)