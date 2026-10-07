"""
apply_features.py
-----------------
Parses implementation_plan.md for selected features, updates preprocessing.py,
and runs train.py to retrain the model with the customized feature set.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_PY = PROJECT_ROOT / "src" / "preprocessing.py"

def parse_selected_features(plan_path: Path) -> tuple[list[str], list[str]]:
    """Parse numerical and categorical features from implementation_plan.md"""
    content = plan_path.read_text(encoding="utf-8")
    
    numerical_cols = []
    categorical_cols = []
    
    # Simple state machine to parse the markdown sections
    current_section = None
    
    for line in content.splitlines():
        line = line.strip()
        if "### Numerical Features" in line:
            current_section = "numerical"
            continue
        elif "### Categorical Features" in line:
            current_section = "categorical"
            continue
        elif line.startswith("## ") or (line.startswith("### ") and "Features" not in line):
            current_section = None
            continue
            
        if current_section and line.startswith("-"):
            # Match checkmarks like "- [x] `Age`" or plain bullets like "- Age" or "- [ ] `Age`"
            # We want to identify the feature name and if it is checked
            is_checked = True
            if "[ ]" in line:
                is_checked = False
                
            # Extract word inside backticks or raw name
            match = re.search(r"`([^`]+)`", line)
            if not match:
                # Try to extract the word after bullet/checkbox
                match = re.search(r"-\s*(?:\[[ xX]\]\s*)?([a-zA-Z0-9_]+)", line)
                
            if match and is_checked:
                feat_name = match.group(1).strip()
                if current_section == "numerical":
                    numerical_cols.append(feat_name)
                elif current_section == "categorical":
                    categorical_cols.append(feat_name)
                    
    return numerical_cols, categorical_cols

def update_preprocessing_py(numerical_cols: list[str], categorical_cols: list[str]) -> None:
    """Rewrite preprocessing.py with the new numerical and categorical columns lists"""
    content = PREPROCESSING_PY.read_text(encoding="utf-8")
    
    # Update NUMERICAL_COLS list block
    num_str = "NUMERICAL_COLS = [\n" + "".join(f"    \"{col}\",\n" for col in numerical_cols) + "]"
    content = re.sub(
        r"NUMERICAL_COLS\s*=\s*\[.*?\]",
        num_str,
        content,
        flags=re.DOTALL
    )
    
    # Update CATEGORICAL_COLS list block
    cat_str = "CATEGORICAL_COLS = [\n" + "".join(f"    \"{col}\",\n" for col in categorical_cols) + "]"
    content = re.sub(
        r"CATEGORICAL_COLS\s*=\s*\[.*?\]",
        cat_str,
        content,
        flags=re.DOTALL
    )
    
    PREPROCESSING_PY.write_text(content, encoding="utf-8")
    print(f"Updated {PREPROCESSING_PY.name} successfully.")
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/apply_features.py <path_to_implementation_plan.md>")
        sys.exit(1)
        
    plan_path = Path(sys.argv[1])
    if not plan_path.exists():
        print(f"Error: Implementation plan not found at {plan_path}")
        sys.exit(1)
        
    print(f"Parsing implementation plan: {plan_path}")
    num_cols, cat_cols = parse_selected_features(plan_path)
    
    if not num_cols and not cat_cols:
        print("Error: No active features parsed from implementation plan.")
        sys.exit(1)
        
    # Update preprocessor file
    update_preprocessing_py(num_cols, cat_cols)
    
    # Import train and run it
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.train import train
    
    print("\nRunning training pipeline...")
    train()
    print("\nTraining completed successfully! App has been updated.")

if __name__ == "__main__":
    main()
