# Copilot Instructions for AI Coding Agents

## Project Overview
- This codebase is for machine learning projects. The structure and conventions may vary by subproject.
- There is no monolithic architecture; expect multiple independent or loosely coupled components.

## Key Conventions
- Place new scripts, notebooks, or modules in clearly named subfolders by topic or experiment.
- Use descriptive names for files and functions. Avoid generic names like `script.py` or `test1.ipynb`.
- Document non-obvious logic with inline comments or markdown cells (for notebooks).
- Prefer standard Python and ML libraries (e.g., numpy, pandas, scikit-learn, matplotlib, tensorflow, pytorch) unless a project-specific dependency is documented.

## Workflows
- There is no global build or test system. Each subproject may have its own workflow.
- For Python scripts, run with `python <script>.py` from the relevant directory.
- For Jupyter notebooks, use VS Code's built-in notebook support.
- If a `requirements.txt` or `environment.yml` exists in a subfolder, install dependencies before running code in that folder.

## Patterns & Examples
- Organize data, models, and results in subfolders: e.g., `data/`, `models/`, `results/`.
- Use versioned filenames for experiments: e.g., `experiment_v1.ipynb`, `model_v2.py`.
- Save intermediate results to disk to avoid recomputation.
- If using custom modules, add the module directory to `sys.path` at the top of scripts/notebooks.

## Integration & External Dependencies
- Check for `README.md` or comments in each subproject for integration details.
- External data sources or APIs should be documented in the relevant script or notebook.

## When in Doubt
- If a pattern or workflow is unclear, prefer explicitness and local documentation.
- Ask for clarification or add a comment describing your uncertainty.
