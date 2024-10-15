import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import anthropic # type: ignore
import os
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from CONFLUENCE.CONFLUENCE import CONFLUENCE # type: ignore

CONFLUENCE_OVERVIEW = """
CONFLUENCE (Community Optimization and Numerical Framework for Large-domain Understanding of Environmental Networks and Computational Exploration) is an integrated hydrological modeling platform. It combines various components for data management, model setup, optimization, uncertainty analysis, forecasting, and visualization across multiple scales and regions.

Key features of CONFLUENCE include:
1. Support for multiple hydrological models (currently SUMMA, FLASH)
2. Flexible spatial discretization (e.g., GRUSs, HRUs, lumped)
3. Various forcing data options (e.g., RDRS, ERA5)
4. Advanced calibration and optimization techniques
6. Geospatial analysis and preprocessing tools
7. Performance evaluation metrics
8. Visualization and reporting capabilities

CONFLUENCE aims to provide a comprehensive, modular, and extensible framework for hydrological modeling, suitable for both research and operational applications.
"""

EXPERT_PROMPTS = {
    "Hydrologist Expert": f"""
    {CONFLUENCE_OVERVIEW}
    
    As the Hydrologist Expert, your role is to analyze the CONFLUENCE model settings with a focus on hydrological processes and model structure. Consider the following aspects in your analysis:
    1. Appropriateness of the chosen hydrological model for the given domain
    2. Representation of key hydrological processes (e.g., surface runoff, infiltration, evapotranspiration)
    3. Temporal and spatial scales of the model setup
    4. Consistency between model structure and the expected dominant hydrological processes in the study area
    5. Potential limitations or assumptions in the model structure that may affect results
    
    Provide insights on how the current configuration might impact the model's ability to accurately represent the hydrological system, and suggest potential improvements or alternative approaches where applicable.
    """,

    "Data Preprocessing Expert": f"""
    {CONFLUENCE_OVERVIEW}
    
    As the Data Preprocessing Expert, your role is to evaluate the data preparation and quality control aspects of the CONFLUENCE setup. Focus on the following areas:
    1. Quality and appropriateness of the chosen forcing dataset
    2. Temporal and spatial resolution of input data
    3. Data gap-filling or interpolation methods, if any
    4. Consistency checks and quality control procedures
    5. Preprocessing steps for model inputs (e.g., unit conversions, aggregation/disaggregation)
    6. Handling of metadata and data provenance
    
    Assess the adequacy of the current data preprocessing approach and suggest any improvements that could enhance data quality or model performance.
    """,

    "Model Calibration Expert": f"""
    {CONFLUENCE_OVERVIEW}
    
    As the Model Calibration Expert, your task is to analyze the calibration and optimization strategy in the CONFLUENCE setup. Consider the following aspects:
    1. Choice of parameters for calibration
    2. Selection of objective function(s) and performance metrics
    3. Calibration algorithm and its configuration
    4. Definition of calibration and validation periods
    5. Approach to handling equifinality and parameter uncertainty
    6. Strategies for avoiding overfitting
    7. Consideration of multiple calibration objectives or sites
    
    Evaluate the effectiveness of the current calibration approach and propose any refinements that could lead to more robust or efficient parameter estimation.
    """,

    "Geospatial Analysis Expert": f"""
    {CONFLUENCE_OVERVIEW}
    
    As the Geospatial Analysis Expert, your focus is on the spatial aspects of the CONFLUENCE model setup. Analyze the following elements:
    1. Method of spatial discretization (e.g., HRUs, grid-based)
    2. Resolution and quality of spatial data inputs (e.g., DEM, land cover, soil maps)
    3. Delineation of catchments or sub-basins
    4. Representation of spatial heterogeneity
    5. Handling of scale issues between data sources and model resolution
    6. Geospatial preprocessing steps and tools used
    7. Consideration of spatial patterns in model inputs and outputs
    
    Assess the appropriateness of the current geospatial setup and suggest any improvements that could enhance the spatial representation in the model.
    """,

    "Performance Metrics Expert": f"""
    {CONFLUENCE_OVERVIEW}
    
    As the Performance Metrics Expert, your role is to evaluate the approach to model performance assessment in the CONFLUENCE setup. Consider the following aspects:
    1. Selection of performance metrics for different hydrological variables
    2. Appropriateness of chosen metrics for the study objectives
    3. Consideration of multiple aspects of model performance (e.g., water balance, timing, low/high flows)
    4. Approach to multi-site or multi-variable performance assessment
    5. Methods for visualizing and communicating model performance
    6. Strategies for identifying and diagnosing model deficiencies
    7. Consideration of uncertainty in performance evaluation
    
    Evaluate the comprehensiveness and effectiveness of the current performance assessment approach, and suggest any additional metrics or methods that could provide a more thorough evaluation of model performance.
    """
}


class AnthropicAPI:
    """A wrapper for the Anthropic API."""

    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_text(self, prompt: str, system_message: str, max_tokens: int = 1000) -> str:
        """Generate text using the Anthropic API."""
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=max_tokens,
            temperature=0,
            system=system_message,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        return message.content[0].text

class Expert:
    def __init__(self, name: str, expertise: str, api: AnthropicAPI):
        self.name = name
        self.expertise = expertise
        self.api = api
        self.prompt = EXPERT_PROMPTS[name]

    def analyze_settings(self, settings: Dict[str, Any], confluence_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        summarized_settings = summarize_settings(settings)
        system_message = f"You are a world-class expert in {self.expertise} with extensive knowledge of the CONFLUENCE model. Provide insightful analysis of the given model settings, focusing on your area of expertise."
        prompt = f"{self.prompt}\n\nAnalyze the following CONFLUENCE model settings, focusing on {self.expertise}. Provide insights and suggestions:\n\n{summarized_settings}"
        
        if confluence_results:
            prompt += f"\n\nCONFLUENCE Results: {confluence_results}"
        
        analysis = self.api.generate_text(prompt, system_message)
        return {"full_analysis": analysis}

class HydrologistExpert(Expert):
    """Expert in hydrological processes and model structure."""

    def __init__(self, api: AnthropicAPI):
        super().__init__("Hydrologist Expert", "hydrological processes and model structure", api)

    def generate_perceptual_model(self, settings: Dict[str, Any]) -> str:
        """Generate a perceptual model summary for the domain being modelled."""
        summarized_settings = summarize_settings(settings)
        system_message = "You are a world-class hydrologist. Create a concise perceptual model summary for the given domain based on the CONFLUENCE model settings."
        prompt = f"Based on the following CONFLUENCE model settings, generate a detailed perceptual model summary for the domain being modelled, citing the relevant literature. Include key hydrological processes and their interactions:\n\n{summarized_settings}"
        perceptual_model = self.api.generate_text(prompt, system_message)
        return perceptual_model

class DataPreprocessingExpert(Expert):
    """Expert in data quality and preprocessing for hydrological models."""

    def __init__(self, api: AnthropicAPI):
        super().__init__("Data Preprocessing Expert", "data quality and preprocessing for hydrological models", api)

class ModelCalibrationExpert(Expert):
    """Expert in parameter estimation and optimization for hydrological models."""

    def __init__(self, api: AnthropicAPI):
        super().__init__("Model Calibration Expert", "parameter estimation and optimization for hydrological models", api)

class GeospatialAnalysisExpert(Expert):
    """Expert in spatial discretization and geofabric setup for hydrological models."""

    def __init__(self, api: AnthropicAPI):
        super().__init__("Geospatial Analysis Expert", "spatial discretization and geofabric setup for hydrological models", api)

class PerformanceMetricsExpert(Expert):
    """Expert in evaluation of hydrological model performance."""

    def __init__(self, api: AnthropicAPI):
        super().__init__("Performance Metrics Expert", "evaluation of hydrological model performance", api)

class Chairperson:
    """Chairperson of the INDRA system, responsible for coordinating experts and generating the final report."""
    def __init__(self, experts: List[Expert], api: AnthropicAPI):
        self.experts = experts
        self.api = api

    def load_control_file(self, file_path: Path) -> Dict[str, Any]:
        """Load the CONFLUENCE control file."""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    def consult_experts(self, settings: Dict[str, Any], confluence_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Consult all experts and gather their analyses."""
        synthesis = {}
        for expert in self.experts:
            synthesis[expert.name] = expert.analyze_settings(settings, confluence_results)
        return synthesis

    def generate_report(self, settings: Dict[str, Any], synthesis: Dict[str, Any], confluence_results: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Generate a comprehensive report based on expert analyses."""
        summarized_settings = summarize_settings(settings)
        
        # Generate panel discussion summary
        system_message = "You are the chairperson of INDRA. Summarize the expert panel discussion based on their analyses."
        prompt = f"Summarize the following expert analyses of CONFLUENCE model settings as if it were a panel discussion:\n\n"
        for expert_name, analysis in synthesis.items():
            prompt += f"{expert_name} Analysis: {analysis['full_analysis']}\n\n"
        
        if confluence_results:
            prompt += f"CONFLUENCE Results: {confluence_results}\n\n"
        
        panel_summary = self.api.generate_text(prompt, system_message, max_tokens=1500)
        
        # Generate concluded summary and suggestions
        system_message = "You are the chairperson of INDRA. Provide a concluded summary of the expert analyses and suggest improvements."
        prompt = f"Based on the following expert analyses, panel discussion, and CONFLUENCE results (if provided), provide a concluded summary for the CONFLUENCE model settings and suggest improvements:\n\nPanel Discussion: {panel_summary}\n\nSettings: {summarized_settings}"
        
        if confluence_results:
            prompt += f"\n\nCONFLUENCE Results: {confluence_results}"
        
        conclusion = self.api.generate_text(prompt, system_message, max_tokens=1500)
        
        # Extract suggestions from the conclusion
        suggestions = self._extract_suggestions(conclusion)
        
        return {
            "concluded_summary": conclusion,
            "panel_summary": panel_summary
        }, suggestions

    def _extract_suggestions(self, conclusion: str) -> Dict[str, Any]:
        """Extract and summarize suggestions from the conclusion."""
        system_message = "You are the chairperson of INDRA, tasked with summarizing suggestions for improving a CONFLUENCE model configuration."
        
        prompt = f"""
        Based on the following conclusion from our expert panel, please summarize the key suggestions for improving the CONFLUENCE model configuration. Present these suggestions as a Python dictionary where the keys are the configuration parameters to be changed, and the values are the suggested new values or approaches.

        Conclusion:
        {conclusion}

        Format your response as follows:

        SUGGESTIONS DICTIONARY:
        suggestions = {{
            "PARAMETER1": "suggested change 1",
            "PARAMETER2": "suggested change 2",
            ...
        }}

        SUMMARY:
        <A brief summary of the suggestions and their potential impact>
        """

        response = self.api.generate_text(prompt, system_message, max_tokens=1000)

        # Extract the suggestions dictionary
        dict_part = response.split("SUGGESTIONS DICTIONARY:")[1].split("SUMMARY:")[0].strip()
        local_vars = {}
        exec(dict_part, globals(), local_vars)
        suggestions = local_vars['suggestions']

        # Extract the summary
        summary = response.split("SUMMARY:")[1].strip()

        print("Suggestions Summary:")
        print(summary)

        return suggestions
    
    def suggest_changes(self, settings: Dict[str, Any], suggestions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print("\nBased on the analysis, here are some suggested changes:")
        for param, suggestion in suggestions.items():
            print(f"{param}: {suggestion}")

        if input("\nWould you like to make any changes? (y/n): ").lower() == 'y':
            updated_settings = settings.copy()
            print("\nCurrent settings:")
            for key, value in updated_settings.items():
                print(f"{key}: {value}")
            while True:
                key = input("\nEnter the setting you'd like to change (or 'done' to finish): ")
                if key.lower() == 'done':
                    break
                if key in updated_settings:
                    value = input(f"Enter the new value for {key} (current: {updated_settings[key]}): ")
                    updated_settings[key] = value
                    print(f"Updated {key} to {value}")
                else:
                    print(f"Setting {key} not found in configuration.")
            return updated_settings
        return None

    def save_updated_config(self, config: Dict[str, Any], file_path: Path):
        with open(file_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Updated configuration saved to {file_path}")

    def expert_initiation(self, watershed_name: str) -> Tuple[Dict[str, Any], str]:
        """Consult experts to determine optimal initial settings for the given watershed."""
        
        system_message = "You are the chairperson of INDRA, coordinating a panel of hydrological modeling experts to determine optimal initial settings for a CONFLUENCE model configuration."
        
        prompt = f"""
        We are initiating a new CONFLUENCE project for the watershed named: {watershed_name}

        As the panel of experts, please suggest optimal initial settings for the following configuration parameters:

        1. HYDROLOGICAL_MODEL (e.g., SUMMA, FLASH)
        2. ROUTING_MODEL (e.g., mizuroute)
        3. FORCING_DATASET (e.g., RDRS, ERA5)
        4. FORCING_START_YEAR and FORCING_END_YEAR
        5. DOMAIN_DISCRETIZATION method (e.g., elevation, soilclass, landclass)
        6. ELEVATION_BAND_SIZE (if using elevation-based discretization)
        7. MIN_HRU_SIZE
        8. POUR_POINT_COORDS coordinates lat/lon to define watershed to delineate must be specified as decimals with 6 digits in the format 'lat/lon'. Select coordinates on the river main step close at estuary or confluence.
        9. BOUNDING_BOX_COORDS coordinates of the bounding box of the watershed must be specified as decimals with 2 digits in the format 'lat_max/lat_min/lon_max/lon_min'

        For each parameter, provide a brief justification for your recommendation.

        After gathering all expert opinions, please:
        1. Create a Python dictionary named 'config' with the agreed-upon settings.
        2. Provide a summary of the justifications for these settings.

        Present your response in the following format:

        CONFIG DICTIONARY:
        config = {{
            "PARAMETER1": value1,
            "PARAMETER2": value2,
            ...
        }}

        JUSTIFICATION SUMMARY:
        <A concise summary of the justifications for the chosen settings>
        """
        
        response = self.api.generate_text(prompt, system_message, max_tokens=1500)
        
        # Split the response into config and justification
        config_part, justification_part = response.split("JUSTIFICATION SUMMARY:")
        
        # Extract the Python dictionary code
        config_code = config_part.split("CONFIG DICTIONARY:")[1].strip()
        
        # Remove any markdown code block syntax
        config_code = config_code.replace("```python", "").replace("```", "").strip()
        
        # Execute the config dictionary code
        local_vars = {}
        exec(config_code, globals(), local_vars)
        config = local_vars['config']
        
        # Clean up the justification summary
        justification_summary = justification_part.strip()
        
        return config, justification_summary

class INDRA:
    """
    Intelligent Network for Dynamic River Analysis (INDRA)

    INDRA is a system that analyzes CONFLUENCE model settings using a panel of expert AI agents.
    It provides comprehensive insights and suggestions for improving hydrological modeling.

    Attributes:
        api (AnthropicAPI): The API for interacting with the Anthropic language model.
        experts (List[Expert]): A list of expert AI agents specialized in various aspects of hydrological modeling.
        chairperson (Chairperson): The chairperson who coordinates the experts and generates the final report.

    Methods:
        run(control_file_path: Path) -> str: Run the INDRA analysis on the given CONFLUENCE control file.
    """

    def __init__(self, api_key):
        self.api = AnthropicAPI(api_key)
        self.experts = [
            HydrologistExpert(self.api),
            DataPreprocessingExpert(self.api),
            ModelCalibrationExpert(self.api),
            GeospatialAnalysisExpert(self.api),
            PerformanceMetricsExpert(self.api)
        ]
        self.chairperson = Chairperson(self.experts, self.api)

    def run(self, control_file_path: Optional[Path] = None, confluence_results: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
        if control_file_path is None:
            control_file_path = self.initiate_new_project()
        
        settings = self.chairperson.load_control_file(control_file_path)
        synthesis = self.chairperson.consult_experts(settings, confluence_results)
        report, suggestions = self.chairperson.generate_report(settings, synthesis, confluence_results)
        
        print("\nINDRA Analysis Summary:")
        print("------------------------")
        print(f"Analyzed config file: {control_file_path}")
        print("\nKey points from the analysis:")
        for i, key_point in enumerate(report['concluded_summary'].split('\n')[:5], 1):  # Print first 5 lines of summary
            print(f"{i}. {key_point}")
        
        print("\nSuggestions for improvement:")
        for param, suggestion in suggestions.items():
            print(f"{param}: {suggestion}")
        
        updated_config = self.chairperson.suggest_changes(settings, suggestions)
        if updated_config:
            self.chairperson.save_updated_config(updated_config, control_file_path)
            print(f"\nUpdated configuration saved to {control_file_path}")
                    
        # Generate perceptual model summary
        hydrologist = next(expert for expert in self.experts if isinstance(expert, HydrologistExpert))
        perceptual_model = hydrologist.generate_perceptual_model(settings)
        
        # Save the report
        report_path = Path(os.getcwd()) / "indra_reports" 
        report_path.mkdir(parents=True, exist_ok=True) 
        report_name = f"{control_file_path.stem}_INDRA_report.txt"

        with open(report_path / report_name, 'w') as f:
            f.write("INDRA Analysis Report\n")
            f.write("=====================\n\n")
            f.write("1. Concluded Summary\n")
            f.write("--------------------\n")
            f.write(report['concluded_summary'])
            f.write("\n\n2. Panel Expert Discussions\n")
            f.write("-----------------------------\n")
            f.write(report['panel_summary'])
            f.write("\n\n3. Perceptual Model Summary\n")
            f.write("-----------------------------\n")
            f.write(perceptual_model)
            f.write("\n\n4. Suggestions\n")
            f.write("---------------\n")
            for suggestion in suggestions:
                f.write(f"- {suggestion}\n")
        
        print(f"INDRA report saved to: {report_path / report_name}")

        if updated_config:
            # Run CONFLUENCE with updated config
            confluence_results = self.run_confluence(control_file_path)
            
            # Analyze CONFLUENCE results
            confluence_analysis = self.analyze_confluence_results(confluence_results)
            print("\nCONFLUENCE Run Analysis:")
            print(confluence_analysis)

        return confluence_analysis
    
    def initiate_new_project(self) -> Path:
        print("Initiating a new CONFLUENCE project.")
        watershed_name = input("Enter the name of the watershed you want to model: ")

        config, justification = self.chairperson.expert_initiation(watershed_name)

        # Add default parameters
        default_params = {
            "DOMAIN_NAME": watershed_name,
            "CONFLUENCE_DATA_DIR": "/Users/darrieythorsson/compHydro/data/CONFLUENCE_data",
            "CONFLUENCE_CODE_DIR": "/Users/darrieythorsson/compHydro/code/CONFLUENCE",
            "POUR_POINT_SHP_PATH": "default",
            "POUR_POINT_SHP_NAME": "default",
            "DOMAIN_DEFINITION_METHOD": "delineate",
            "GEOFABRIC_TYPE": "TDX",
            "STREAM_THRESHOLD": 5000,
            "LUMPED_WATERSHED_METHOD": "pysheds",
            "CLEANUP_INTERMEDIATE_FILES": True,
            "DELINEATE_BY_POURPOINT": True,
            "OUTPUT_BASINS_PATH": "default",
            "OUTPUT_RIVERS_PATH": "default",
            "DEM_PATH": "default",
            "DEM_NAME": "elevation.tif",
            "SOURCE_GEOFABRIC_BASINS_PATH": "/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_NorthAmerica/shapefiles/catchment/7020021430-basins.gpkg",
            "SOURCE_GEOFABRIC_RIVERS_PATH": "/Users/darrieythorsson/compHydro/data/CWARHM_data/domain_NorthAmerica/shapefiles/river_network/7020021430-streamnet.gpkg",
            "TAUDEM_DIR": "default",
            "OUTPUT_DIR": "default",
            "CATCHMENT_PLOT_DIR": "default"


        }
        
        # Update the config with default parameters
        config.update(default_params)
        config_path = Path("0_config_files") 
        config_path.mkdir(parents=True, exist_ok=True) 
        config_name = "config_active.yaml"
        with open(config_path / config_name, 'w') as f:
            yaml.dump(config, f)
        
        print(f"Initial configuration saved to {config_path / config_name}")
        print("\nJustification for the chosen settings:")
        print(justification)
        
        return config_path / config_name
    
    def run_confluence(self, config_path: Path) -> Dict[str, Any]:
        """
        Run CONFLUENCE with the given configuration file.

        Args:
            config_path (Path): Path to the configuration file.

        Returns:
            Dict[str, Any]: Results from the CONFLUENCE run.
        """
        confluence = CONFLUENCE(config_path)  # Initialize CONFLUENCE

        try:            
            # Run CONFLUENCE
            confluence.run_workflow()
            
            # Get the results
            confluence_results = self.confluence.get_results()
            
            return confluence_results
        
        except Exception as e:
            print(f"Error running CONFLUENCE: {str(e)}")
            return {"error": str(e)}
        
    def analyze_confluence_results(self, confluence_results: Dict[str, Any]) -> str:
        """
        Analyze the results from a CONFLUENCE run.

        Args:
            confluence_results (Dict[str, Any]): Results from CONFLUENCE run.

        Returns:
            str: Analysis of the CONFLUENCE results.
        """
        system_message = "You are an expert in analyzing hydrological model results."
        
        prompt = f"""
        Please analyze the following results from a CONFLUENCE model run:

        {confluence_results}

        Provide a brief summary of the model performance, highlighting any notable aspects or potential issues.
        """

        analysis = self.api.generate_text(prompt, system_message, max_tokens=500)
        return analysis
    
    
 
def summarize_settings(settings: Dict[str, Any], max_length: int = 2000) -> str:
    """Summarize the settings to a maximum length."""
    settings_str = yaml.dump(settings)
    if len(settings_str) <= max_length:
        return settings_str
    
    summarized = "Settings summary (truncated):\n"
    for key, value in settings.items():
        summary = f"{key}: {str(value)[:100]}...\n"
        if len(summarized) + len(summary) > max_length:
            break
        summarized += summary
    
    return summarized


# Usage
if __name__ == "__main__":

    indra = INDRA(api_key)

    # Ask user if they want to use an existing config or create a new one
    use_existing = input("Do you want to use an existing config file? (y/n): ").lower() == 'y'
    
    if use_existing:
        control_file_path = control_file_path = Path("/Users/darrieythorsson/compHydro/code/INDRA/0_config_files/config_active.yaml")
    else:
        control_file_path = None  # This will trigger the creation of a new config
    
    confluence_results = None  # Replace with actual results if available
    confluence_analysis = indra.run(control_file_path, confluence_results)