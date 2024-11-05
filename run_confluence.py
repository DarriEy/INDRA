#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from CONFLUENCE.CONFLUENCE import CONFLUENCE # type: ignore
import argparse

def run_confluence(config_path: Path) -> dict:
    """
    Run CONFLUENCE workflow with given configuration file.
    
    Args:
        config_path (Path): Path to CONFLUENCE configuration file
        
    Returns:
        dict: Results from CONFLUENCE run or error message
    """
    try:
        # Initialize and run CONFLUENCE
        confluence = CONFLUENCE(config_path)
        confluence.run_workflow()
        
        return {"status": "completed", "message": f"CONFLUENCE run completed successfully for config: {config_path}"}
        
    except Exception as e:
        return {"status": "failed", "error": str(e)}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run CONFLUENCE workflow')
    parser.add_argument('config_path', type=str, help='Path to CONFLUENCE configuration file')
    args = parser.parse_args()
    
    # Run CONFLUENCE
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
        
    results = run_confluence(config_path)
    
    # Print results
    if results['status'] == 'completed':
        print(results['message'])
        sys.exit(0)
    else:
        print(f"Error running CONFLUENCE: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()