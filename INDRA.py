import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import anthropic # type: ignore
import os
import sys
import subprocess
import time

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

    "Data Science Expert": f"""
    {CONFLUENCE_OVERVIEW}
    
    As the Data Science Expert, your role is to evaluate the data preparation and quality control aspects of the CONFLUENCE setup. Focus on the following areas:
    1. Quality and appropriateness of the chosen forcing dataset
    2. Temporal and spatial resolution of input data
    
    Assess the adequacy of the current data preprocessing approach and suggest any improvements that could enhance data quality or model performance.
    """,

    "Hydrogeology Expert": f"""
    {CONFLUENCE_OVERVIEW}
    
    As the Hydrogeology Expert, your role is to analyze the CONFLUENCE model settings with a focus on hydrogeological processes and model structure. Consider the following aspects in your analysis:
    1. Appropriateness of the chosen hydrological model for the given domain
    2. Representation of key hydrogeological processes (e.g., surface runoff, infiltration, evapotranspiration)
    3. Temporal and spatial scales of the model setup
    4. Consistency between model structure and the expected dominant hydrogeological processes in the study area
    5. Potential limitations or assumptions in the model structure that may affect results
    
     Provide insights on how the current configuration might impact the model's ability to accurately represent the hydrogeologicallogical system, and suggest potential improvements or alternative approaches where applicable.
    """,

    "Meteorological Expert": f"""
    {CONFLUENCE_OVERVIEW}
    
    As the Meteorological Expert, your role is to analyze the CONFLUENCE model settings with a focus on meteorological processes and model structure. Consider the following aspects in your analysis:
    1. Quality and appropriateness of the chosen forcing dataset
    2. Representation of key meteorological processes (e.g., surface runoff, infiltration, evapotranspiration)
    3. Temporal and spatial scales of the model setup
    
     Provide insights on how the current configuration might impact the model's ability to accurately represent the hydrogeologicallogical system, and suggest potential improvements or alternative approaches where applicable.
    """
}


class AnthropicAPI:
    """A wrapper for the Anthropic API."""

    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_text(self, prompt: str, system_message: str, max_tokens: int = 1750) -> str:
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
        prompt = f'''Based on the following CONFLUENCE model domain, generate a detailed and extensive perceptual model summary for the domain being modelled, 
                     citing the relevant literature and providing a list of references. Include key hydrological processes and their interaction. 
                     Summarize previous modelling efforts in this basin and their findings. Identify modelling approaches that have provided good results or 
                     are likely to provide good results. Also identify (if available in the literature) modelling approaches that have not proven fruitful.
                     :\n\n{summarized_settings}'''
        perceptual_model = self.api.generate_text(prompt, system_message)
        return perceptual_model

class DataScienceExpert(Expert):
    """Expert in data science and preprocessing for hydrological models."""

    def __init__(self, api: AnthropicAPI):
        super().__init__("Data Science Expert", "data science and preprocessing for hydrological models", api)

class HydrogeologyExpert(Expert):
    """Expert in parameter estimation and optimization for hydrological models."""

    def __init__(self, api: AnthropicAPI):
        super().__init__("Hydrogeology Expert", "parameter estimation and optimization for hydrological models", api)
    
    def generate_perceptual_model(self, settings: Dict[str, Any]) -> str:
        """Generate a perceptual model summary for the domain being modelled."""
        summarized_settings = summarize_settings(settings)
        system_message = "You are a world-class hydrogeologist. Create a concise perceptual model summary for the given domain based on the CONFLUENCE model settings."
        prompt = f'''Based on the following CONFLUENCE model domain, generate a detailed and extensive perceptual model summary for the domain being modelled, 
                     citing the relevant literature and providing a list of references. Include key hydrogeolological processes and their interaction. 
                     Summarize previous modelling efforts in this basin and their findings. Identify modelling approaches that have provided good results or 
                     are likely to provide good results. Also identify (if available in the literature) modelling approaches that have not proven fruitful.
                     :\n\n{summarized_settings}'''
        perceptual_model = self.api.generate_text(prompt, system_message)
        return perceptual_model

class MeteorologicalExpert(Expert):
    """Expert in evaluation of hydrological model performance."""

    def __init__(self, api: AnthropicAPI):
        super().__init__("Meteorological Expert", "evaluation of hydrological model performance", api)
    
    def generate_perceptual_model(self, settings: Dict[str, Any]) -> str:
        """Generate a perceptual model summary for the domain being modelled."""
        summarized_settings = summarize_settings(settings)
        system_message = "You are a world-class meteorological. Create a concise perceptual model summary for the given domain based on the CONFLUENCE model settings."
        prompt = f'''Based on the following CONFLUENCE model domain, generate a detailed and extensive perceptual model summary for the domain being modelled, 
                     citing the relevant literature and providing a list of references. Include key meteorological processes and their interaction. 
                     Summarize previous modelling efforts in this basin and their findings. Identify modelling approaches that have provided good results or 
                     are likely to provide good results. Also identify (if available in the literature) modelling approaches that have not proven fruitful.
                     :\n\n{summarized_settings}'''
        perceptual_model = self.api.generate_text(prompt, system_message)
        return perceptual_model

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
        
        system_message = '''You are the chairperson of INDRA, coordinating a panel of hydrological modeling experts 
                            to determine optimal initial settings for a CONFLUENCE model configuration.'''
        
        prompt = f"""
        We are initiating a new CONFLUENCE project for the watershed named: {watershed_name}

        As the panel of experts, please suggest optimal initial settings for the following configuration parameters:

        1. HYDROLOGICAL_MODEL (options: SUMMA, FLASH)
        2. ROUTING_MODEL (options: mizuroute)
        3. FORCING_DATASET (options: RDRS, ERA5)
        4. DOMAIN_DISCRETIZATION method (options, elevation, soilclass, landclass)
        5. ELEVATION_BAND_SIZE (if using elevation-based discretization)
        6. MIN_HRU_SIZE: Minimum size of the model domain HRUs, in km2 recommended 10 km2 for large watersheds and 1 km2 for small watersheds
        7. POUR_POINT_COORDS: coordinates lat/lon to define watershed to delineate must be specified as decimals with 6 digits 
                             in the format 'lat/lon'. Select coordinates on the river main step close at estuary or confluence.
        8. BOUNDING_BOX_COORDS: coordinates of the bounding box of the watershed must be specified as decimals with 2 digits 
                               in the format 'lat_max/lon_min/lat_min/lon_max.'

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

    def __init__(self):
        """
        Initialize INDRA with API key from system environment variables.
        Raises:
            ValueError: If the required API key is not found in environment variables.
        """
        # Get API key from system environment
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in system environment variables. "
                "Please ensure the API key is properly set in your system PATH."
            )

        self.api = AnthropicAPI(api_key)
        self.experts = [
            HydrologistExpert(self.api),
            DataScienceExpert(self.api),
            HydrogeologyExpert(self.api),
            MeteorologicalExpert(self.api)
        ]
        self.chairperson = Chairperson(self.experts, self.api)

    def _generate_perceptual_models(self, watershed_name: str) -> Dict[str, str]:
        """
        Generate perceptual models from each domain expert.
        
        Args:
            watershed_name (str): Name of the watershed being modeled.
        
        Returns:
            Dict[str, str]: Dictionary containing perceptual models from each expert.
        """
        print("Consulting domain experts for perceptual model generation...")
        
        perceptual_models = {}
        domain_experts = [expert for expert in self.experts 
                        if isinstance(expert, (HydrologistExpert, HydrogeologyExpert, MeteorologicalExpert))]
        
        settings = {"DOMAIN_NAME": watershed_name}  # Minimal settings for perceptual model generation
        
        for expert in domain_experts:
            print(f"\nGenerating {expert.name} perceptual model...")
            perceptual_models[expert.name] = expert.generate_perceptual_model(settings)
        
        return perceptual_models

    def _save_perceptual_models(self, file_path: Path, perceptual_models: Dict[str, str]):
        """
        Save perceptual models to a formatted text file.
        
        Args:
            file_path (Path): Path to save the perceptual models.
            perceptual_models (Dict[str, str]): Dictionary of perceptual models from each expert.
        """
        with open(file_path, 'w') as f:
            f.write("INDRA Perceptual Models Report\n")
            f.write("=============================\n\n")
            
            for expert_name, model in perceptual_models.items():
                f.write(f"{expert_name} Perceptual Model\n")
                f.write("-" * (len(expert_name) + 17) + "\n")
                f.write(model)
                f.write("\n\n")

    def _save_synthesis_report(self, report: Dict[str, str], suggestions: Dict[str, Any], 
                          watershed_name: str, report_path: Path):
        """
        Save the synthesis report for an existing project.
        
        Args:
            report (Dict[str, str]): The generated report
            suggestions (Dict[str, Any]): Configuration suggestions
            watershed_name (str): Name of the watershed
            report_path (Path): Directory to save the report
        """
        synthesis_file = report_path / f"synthesis_report_{watershed_name}.txt"
        
        with open(synthesis_file, 'w') as f:
            f.write(f"INDRA Synthesis Report for {watershed_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Expert Panel Discussion Summary\n")
            f.write("-" * 28 + "\n")
            f.write(report['panel_summary'])
            f.write("\n\n")
            
            f.write("Concluded Summary\n")
            f.write("-" * 16 + "\n")
            f.write(report['concluded_summary'])
            f.write("\n\n")
            
            f.write("Configuration Suggestions\n")
            f.write("-" * 24 + "\n")
            for param, suggestion in suggestions.items():
                f.write(f"{param}: {suggestion}\n")
        
        print(f"\nSynthesis report saved to: {synthesis_file}")

    def run(self, control_file_path: Optional[Path] = None, confluence_results: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Run the INDRA analysis with perceptual model generation as first step.
        
        Args:
            control_file_path (Optional[Path]): Path to existing config file. If None, initiates new project.
            confluence_results (Optional[Dict[str, Any]]): Results from previous CONFLUENCE run if available.
        
        Returns:
            Tuple[Dict[str, str], Dict[str, Any]]: Analysis results and suggestions
        """
        # Step 1: Get watershed name and determine if this is a new project
        is_new_project = control_file_path is None
        
        if is_new_project:
            print("Initiating a new CONFLUENCE project.")
            watershed_name = input("Enter the name of the watershed you want to model: ")
        else:
            settings = self.chairperson.load_control_file(control_file_path)
            watershed_name = settings.get('DOMAIN_NAME')

        # Step 2: Generate and save domain perceptual models
        print("\nGenerating perceptual models for the domain...")
        perceptual_models = self._generate_perceptual_models(watershed_name)
        
        # Create report directory if it doesn't exist
        report_path = Path(os.getcwd()) / "indra_reports"
        report_path.mkdir(parents=True, exist_ok=True)
        
        # Save perceptual models to file
        perceptual_model_file = report_path / f"perceptual_model_{watershed_name}.txt"
        self._save_perceptual_models(perceptual_model_file, perceptual_models)
        
        print(f"\nPerceptual models have been generated and saved to: {perceptual_model_file}")
        
        # Prompt user to continue
        if input("\nWould you like to continue with configuration process? (y/n): ").lower() != 'y':
            print("Workflow stopped after perceptual model generation.")
            return {}, {}

        if is_new_project:
            # Handle new project initialization
            control_file_path, expert_config = self._initialize_new_project(watershed_name)
            print("\nNew project initialization completed.")
            
            # Load the initial configuration
            settings = self.chairperson.load_control_file(control_file_path)
            
            # Allow user to modify initial configuration
            if input("\nWould you like to modify the INDRA-suggested configuration? (y/n): ").lower() == 'y':
                updated_settings = self._modify_configuration(settings, expert_config)
                if updated_settings:
                    self._create_config_file_from_template(
                        template_path=Path(__file__).parent / '0_config_files' / 'config_template.yaml',
                        output_path=control_file_path,
                        watershed_name=watershed_name,
                        expert_config={k: v for k, v in updated_settings.items() if k in expert_config}
                    )
                    settings = updated_settings
                    print(f"\nUpdated configuration saved to {control_file_path}")
            
            # Run CONFLUENCE with initial configuration and wait for completion
            print("\nRunning CONFLUENCE with initial configuration...")
            job_info = self.run_confluence(control_file_path)
            
            if 'error' in job_info:
                print(f"Error submitting CONFLUENCE job: {job_info['error']}")
                return {}, {}
            
            # Wait for job completion
            job_id = job_info['job_id']
            print(f"\nWaiting for CONFLUENCE job {job_id} to complete...")
            
            while True:
                status = self._check_job_status(job_id)
                if status.startswith("COMPLETED"):
                    print(f"\nCONFLUENCE job {job_id} completed successfully")
                    break
                elif status.startswith("FAILED") or status.startswith("CANCELLED"):
                    print(f"\nCONFLUENCE job {job_id} {status.lower()}")
                    return {}, {}
                elif status.startswith("PENDING") or status.startswith("RUNNING"):
                    print(f"\rJob status: {status}", end='', flush=True)
                    time.sleep(60)  # Check every minute
                else:
                    print(f"\nUnexpected job status: {status}")
                    return {}, {}
            
            # Read CONFLUENCE results
            confluence_results = self._read_confluence_results(control_file_path, job_id)
            
            if confluence_results:
                confluence_analysis = self.analyze_confluence_results(confluence_results)
                print("\nCONFLUENCE Run Analysis:")
                print(confluence_analysis)
                return confluence_analysis
                
            return {}, {}
        
        else:
            # Rest of the code for existing project remains the same...
            pass

    def _check_job_status(self, job_id: str) -> str:
        """
        Check the status of a SLURM job.
        
        Args:
            job_id (str): SLURM job ID
            
        Returns:
            str: Job status (PENDING, RUNNING, COMPLETED, FAILED, etc.)
        """
        try:
            cmd = f"sacct -j {job_id} --format=State --noheader --parsable2"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                # Get the last status (most recent) and strip whitespace
                status = result.stdout.strip().split('\n')[0]
                return status
            return "UNKNOWN"
        except Exception as e:
            self.logger.error(f"Error checking job status: {str(e)}")
            return "UNKNOWN"

    def _read_confluence_results(self, config_path: Path, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Read the results from a completed CONFLUENCE run.
        
        Args:
            config_path (Path): Path to the configuration file
            job_id (str): SLURM job ID
            
        Returns:
            Optional[Dict[str, Any]]: CONFLUENCE results if available
        """
        try:
            # Read the configuration to get output paths
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get the output directory from config
            domain_name = config.get('DOMAIN_NAME')
            experiment_id = config.get('EXPERIMENT_ID')
            confluence_data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
            
            # Construct paths to key output files
            output_dir = confluence_data_dir / f"domain_{domain_name}/simulations/{experiment_id}"
            
            # Check if output directory exists
            if not output_dir.exists():
                self.logger.error(f"CONFLUENCE output directory not found: {output_dir}")
                return None
                
            # Read key output files and compile results
            results = {
                "output_dir": str(output_dir),
                "job_id": job_id,
                "status": "completed"
            }
            
            # Add any additional results parsing as needed
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error reading CONFLUENCE results: {str(e)}")
            return None
    
    def _modify_configuration(self, settings: Dict[str, Any], expert_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Allow user to modify only INDRA-suggested configuration settings interactively.
        
        Args:
            settings (Dict[str, Any]): Current configuration settings
            expert_config (Dict[str, Any]): Expert-suggested configurations
            
        Returns:
            Optional[Dict[str, Any]]: Modified settings if changes were made, None otherwise
        """
        updated_settings = settings.copy()
        
        print("\nINDRA-suggested configuration settings:")
        modifiable_settings = {k: v for k, v in updated_settings.items() if k in expert_config}
        
        for key, value in modifiable_settings.items():
            print(f"{key}: {value}")
        
        while True:
            print("\nEnter the setting key you'd like to modify (or 'done' to finish, 'cancel' to discard changes):")
            key = input().strip()
            
            if key.lower() == 'done':
                return updated_settings
            elif key.lower() == 'cancel':
                return None
            
            if key in modifiable_settings:
                current_value = updated_settings[key]
                print(f"Current value: {current_value}")
                print(f"Enter new value for {key}:")
                new_value = input().strip()
                
                # Try to preserve the type of the original value
                try:
                    if isinstance(current_value, bool):
                        new_value = new_value.lower() in ('true', 'yes', '1', 'on')
                    elif isinstance(current_value, int):
                        new_value = int(new_value)
                    elif isinstance(current_value, float):
                        new_value = float(new_value)
                    elif isinstance(current_value, str) and ' ' in new_value:
                        new_value = f"'{new_value}'"
                except ValueError:
                    print(f"Warning: Could not convert value to type {type(current_value).__name__}, storing as string")
                
                updated_settings[key] = new_value
                print(f"Updated {key} to: {new_value}")
            else:
                print(f"Setting '{key}' is not an INDRA-suggested configuration and cannot be modified.")
                print("Modifiable settings are:", ', '.join(modifiable_settings.keys()))
        
        return updated_settings
                
    def _initialize_new_project(self, watershed_name: str) -> Tuple[Path, Dict[str, Any]]:
        """
        Initialize a new CONFLUENCE project with expert-suggested configuration.
        Uses a template config file as base and updates it with expert suggestions.
        
        Args:
            watershed_name (str): Name of the watershed being modeled.
        
        Returns:
            Tuple[Path, Dict[str, Any]]: Path to the created configuration file and expert config dictionary
        """
        # Get expert suggestions for the configuration
        expert_config, justification = self.chairperson.expert_initiation(watershed_name)
        
        # Save the justification to a file
        report_path = Path(os.getcwd()) / "indra_reports"
        report_path.mkdir(parents=True, exist_ok=True)
        rationale_file = report_path / f"initial_decision_rationale_{watershed_name}.txt"
        print(justification)
        
        with open(rationale_file, 'w') as f:
            f.write(f"INDRA Initial Configuration Decisions for {watershed_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Expert-Suggested Configuration Parameters:\n")
            f.write("-" * 35 + "\n")
            for key, value in expert_config.items():
                f.write(f"{key}: {value}\n")
            f.write("\nJustification:\n")
            f.write("-" * 13 + "\n")
            f.write(justification)
        
        print(f"\nConfiguration rationale saved to: {rationale_file}")
        
        try:
            # Create config directory
            config_path = Path("0_config_files")
            config_path.mkdir(parents=True, exist_ok=True)
            config_file_path = config_path / f"config_{watershed_name}.yaml"
            print(config_file_path)
            # Load and process template while preserving structure and comments
            self._create_config_file_from_template(
                template_path=Path(__file__).parent / '0_config_files' / 'config_template.yaml',
                output_path=config_file_path,
                watershed_name=watershed_name,
                expert_config=expert_config
            )
            print('config file created from template')
            # Create symbolic link to config_active.yaml
            active_config_path = config_path / "config_active.yaml"
            if active_config_path.exists():
                active_config_path.unlink()
            active_config_path.symlink_to(config_file_path.name)
            print('symlink created')
            return config_file_path, expert_config  # Now returning both values

        except Exception as e:
            print(f"Error initializing new project configuration: {str(e)}")
            raise
    
    def _create_config_file(self, template_config: Dict[str, Any], expert_config: Dict[str, Any], 
                       output_path: Path, header: str = ''):
        """
        Helper method to create a new configuration file with proper formatting.
        
        Args:
            template_config (Dict[str, Any]): Base configuration from template
            expert_config (Dict[str, Any]): Expert-suggested configuration
            output_path (Path): Path to save the new configuration file
            header (str): Header comments to preserve from template
        """
        # Update template with expert configurations
        final_config = template_config.copy()
        for key, value in expert_config.items():
            if key in final_config:
                final_config[key] = value
        
        # Write configuration file
        with open(output_path, 'w') as f:
            # Write header if provided
            if header:
                f.write(header)
                f.write('\n')
            
            # Write each section maintaining the template structure
            current_section = None
            for key, value in final_config.items():
                # Check if this is a new section
                if '### ===' in key:
                    current_section = key
                    f.write(f"\n{key}\n")
                    continue
                
                # Write the configuration entry
                if isinstance(value, str):
                    f.write(f"{key}: {value}  # Original template comment preserved\n")
                else:
                    f.write(f"{key}: {value}\n")

    def _create_config_file_from_template(self, template_path: Path, output_path: Path, 
                                    watershed_name: str, expert_config: Dict[str, Any]):
        """
        Create a new configuration file from template while preserving structure and comments.
        
        Args:
            template_path (Path): Path to template file
            output_path (Path): Path to save new config file
            watershed_name (str): Name of the watershed
            expert_config (Dict[str, Any]): Expert-suggested configurations
        """
        if not template_path.exists():
            raise FileNotFoundError(f"Configuration template not found at: {template_path}")
        
        # Read template file preserving all lines
        with open(template_path, 'r') as f:
            template_lines = f.readlines()
        
        # Update expert config with domain name
        expert_config['DOMAIN_NAME'] = watershed_name
        
        # Process template line by line
        with open(output_path, 'w') as f:
            current_line = ''
            
            for line in template_lines:
                # Preserve comment lines and section headers
                if line.strip().startswith('#') or line.strip().startswith('### ==='):
                    f.write(line)
                    continue
                    
                # Process configuration lines
                if ':' in line:
                    key = line.split(':')[0].strip()
                    if key in expert_config:
                        # Preserve any inline comments
                        comment = line.split('#')[1].strip() if '#' in line else ''
                        value = expert_config[key]
                        if isinstance(value, str):
                            value = f"'{value}'" if ' ' in value else value
                        new_line = f"{key}: {value}"
                        if comment:
                            new_line += f"  # {comment}"
                        f.write(new_line + '\n')
                    else:
                        # Keep original line for non-expert configs
                        f.write(line)
                else:
                    f.write(line)
    '''
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
    ''' 

    def run_confluence(self, config_path: Path) -> Dict[str, Any]:
        """
        Run CONFLUENCE with the given configuration file using SLURM batch system.

        Args:
            config_path (Path): Path to the configuration file.

        Returns:
            Dict[str, Any]: Results from the CONFLUENCE run.
        """
        try:
            # Read the configuration file to get necessary parameters
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Create SLURM submission script
            slurm_script = self._create_slurm_script(config_path, config)
            
            # Submit job
            print(slurm_script)
            #submit_cmd = f"sbatch {slurm_script}"
            submit_cmd = "sbatch run_confluence_batch.sh"
            result = subprocess.run(submit_cmd, shell=True, check=True, capture_output=True, text=True)
            
            # Extract job ID
            job_id = result.stdout.strip().split()[-1]
            print(f"Submitted CONFLUENCE job with ID: {job_id}")
            
            return {"job_id": job_id, "status": "submitted"}
            
        except subprocess.CalledProcessError as e:
            print(f"Error submitting CONFLUENCE job: {str(e)}")
            return {"error": str(e)}

    def _create_slurm_script(self, config_path: Path, config: Dict[str, Any]) -> Path:
        """
        Create a SLURM submission script for CONFLUENCE.
        
        Args:
            config_path (Path): Path to CONFLUENCE configuration file
            config (Dict[str, Any]): Configuration dictionary containing parameters
            
        Returns:
            Path: Path to created SLURM script
        """
        script_path = config_path.parent.parent / "run_confluence_batch.sh"
        
        # Extract parameters from config
        domain_name = config.get('DOMAIN_NAME', 'unknown_domain')
        mpi_processes = config.get('MPI_PROCESSES', 1)
        tool_account = config.get('TOOL_ACCOUNT', '')
        
        script_content = f"""#!/bin/bash
    #SBATCH --job-name=CONFLUENCE_{domain_name}
    #SBATCH --ntasks={mpi_processes}
    #SBATCH --output=CONFLUENCE_single_%j.log
    #SBATCH --error=CONFLUENCE_single_%j.err
    #SBATCH --time=20:00:00
    #SBATCH --mem-per-cpu=5G
    """

        if tool_account:
            script_content += f"#SBATCH --account={tool_account}\n"

        script_content += f"""
    # Load required modules
    module restore confluence_modules

    source {str(Path(config['CONFLUENCE_CODE_DIR']).parent)}/confluence_env/bin/activate

    # Run CONFLUENCE script
    python ../CONFLUENCE/CONFLUENCE.py --config {str(Path(config['CONFLUENCE_CODE_DIR']).parent)}/INDRA/0_config_files/config_active.yaml
    """

        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_path.chmod(0o755)
                
        return script_path
    
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
    try:
        indra = INDRA()
        
        use_existing = input("Do you want to use an existing config file? (y/n): ").lower() == 'y'
        
        if use_existing:
            while True:
                print("\nEnter the path to your configuration file:")
                print("(You can use absolute path or relative path from current directory)")
                control_file_input = input().strip()
                
                # Convert string to Path object
                control_file_path = Path(control_file_input).resolve()
                
                # Validate the path
                if not control_file_path.exists():
                    print(f"\nError: File not found: {control_file_path}")
                    retry = input("Would you like to try another path? (y/n): ").lower() == 'y'
                    if not retry:
                        print("Exiting program.")
                        sys.exit()
                elif not control_file_path.suffix in ['.yaml', '.yml']:
                    print(f"\nError: File must be a YAML file (.yaml or .yml)")
                    retry = input("Would you like to try another path? (y/n): ").lower() == 'y'
                    if not retry:
                        print("Exiting program.")
                        sys.exit()
                else:
                    try:
                        # Validate YAML format
                        with open(control_file_path, 'r') as f:
                            yaml.safe_load(f)
                        print(f"\nUsing configuration file: {control_file_path}")
                        break
                    except yaml.YAMLError:
                        print(f"\nError: Invalid YAML format in {control_file_path}")
                        retry = input("Would you like to try another path? (y/n): ").lower() == 'y'
                        if not retry:
                            print("Exiting program.")
                            sys.exit()
        else:
            control_file_path = None  # This will trigger the creation of a new config
        
        confluence_results = None  # Replace with actual results if available
        confluence_analysis = indra.run(control_file_path, confluence_results)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo set up your API key in the system environment:")
        print("\nFor Unix-like systems (Linux/Mac):")
        print("1. Add this line to your ~/.bashrc or ~/.zshrc:")
        print('   export ANTHROPIC_API_KEY="your-api-key-here"')
        print("2. Run: source ~/.bashrc (or source ~/.zshrc)")
        print("\nFor Windows:")
        print("1. Open System Properties -> Advanced -> Environment Variables")
        print("2. Add a new User Variable:")
        print("   Name: ANTHROPIC_API_KEY")
        print("   Value: your-api-key")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Error details: {str(e)}")