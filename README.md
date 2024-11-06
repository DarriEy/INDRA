# INDRA: Intelligent Network for Dynamic River Analysis

INDRA is an AI-powered system for running the CONFLUENCE workflow using a panel of expert AI agents. It provides comprehensive insights and suggestions for improving hydrological modeling through the integration of multiple expert perspectives
## Features

- Generates perceptual models for new watersheds
- Provides expert-guided configuration for CONFLUENCE setups
- Performs comprehensive model analysis through multiple expert lenses
- Offers configuration suggestions based on expert consensus
- Creates detailed reports for model decisions and their justification

## Setup and Installation

1. Clone the repository:
```
git clone https://github.com/DarriEy/INDRA.git
cd INDRA
```

2. Set up your environment:


Create and activate a Python virtual environment
Install dependencies:
```
pip install -r requirements.txt
```

3. Set up your API key:
```
# For Unix-like systems (Linux/Mac)
export ANTHROPIC_API_KEY="your-api-key-here"

# For Windows (via Command Prompt)
set ANTHROPIC_API_KEY=your-api-key-here
```

## Usage
INDRA can be used in two main ways:

1. New Project Initialization
```
python INDRA.py
# Select 'n' when asked about existing config
# Follow the interactive prompts
```

2. Existing Project Analysis
```
python INDRA.py
# Select 'y' when asked about existing config
# Provide path to your CONFLUENCE configuration file
```

## Dependencies

INDRA is designed to work with the CONFLUENCE workflow. 
See https://github.com/DarriEy/CONFLUENCE for the CONFLUENCE specific dependencies.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.


## Contact

Darri Eythorsson
University of Calgary
darri.eythorsson@ucalgary.ca

## Citation

If you use INDRA in your research, please cite:
[Add paper citation once published]