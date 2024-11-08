import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests # type: ignore
import json
import os
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from CONFLUENCE.CONFLUENCE import CONFLUENCE # type: ignore
TEMPLATE_CONFIG_PATH = Path(__file__).parent / '0_config_files' / 'config_template.yaml'

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

class AnvilGPTAPI:
    """A wrapper for the Anvil GPT API."""
    
    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.url = "https://anvilgpt.rcac.purdue.edu/ollama/api/chat"
        
    def generate_text(self, prompt: str, system_message: str, max_tokens: int = 1750) -> str:
        """Generate text using the Anvil GPT API."""
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        body = {
            "model": "llama3.1:latest",
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=body)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        if 'message' in json_response:
                            content = json_response['message'].get('content', '')
                            full_response += content
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        continue
                        
            return full_response.strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"AnvilGPT API error: {str(e)}")

class Expert:
    def __init__(self, name: str, expertise: str, api: AnvilGPTAPI):
        self.name = name
        self.expertise = expertise
        self.api = api
        self.prompt = EXPERT_PROMPTS[name]

    def analyze_settings(self, settings: Dict[str, Any], confluence_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        summarized_settings = summarize_settings(settings)
        system_message = f"You are a world-class expert in {self.expertise} with extensive knowledge of the CONFLUENCE model."
        prompt = f"{self.prompt}\n\nAnalyze the following CONFLUENCE model settings:\n\n{summarized_settings}"
        
        if confluence_results:
            prompt += f"\n\nCONFLUENCE Results: {confluence_results}"
        
        analysis = self.api.generate_text(prompt, system_message)
        return {"full_analysis": analysis}

class HydrologistExpert(Expert):
    def __init__(self, api: AnvilGPTAPI):
        super().__init__("Hydrologist Expert", "hydrological processes and model structure", api)

    def generate_perceptual_model(self, settings: Dict[str, Any]) -> str:
        summarized_settings = summarize_settings(settings)
        system_message = "You are a world-class hydrologist."
        prompt = f'''Based on the following CONFLUENCE model domain, generate a detailed and extensive perceptual model summary for the domain being modelled, 
                     citing the relevant literature and providing a list of references. Include key  processes and their interaction. 
                     Summarize previous modelling efforts in this basin and their findings. Identify modelling approaches that have provided good results or 
                     are likely to provide good results. Also identify (if available in the literature) modelling approaches that have not proven fruitful.:\n\n{summarized_settings}'''
        perceptual_model = self.api.generate_text(prompt, system_message)
        return perceptual_model

class DataScienceExpert(Expert):
    def __init__(self, api: AnvilGPTAPI):
        super().__init__("Data Science Expert", "data science and preprocessing for hydrological models", api)

class HydrogeologyExpert(Expert):
    def __init__(self, api: AnvilGPTAPI):
        super().__init__("Hydrogeology Expert", "parameter estimation and optimization for hydrological models", api)
    
    def generate_perceptual_model(self, settings: Dict[str, Any]) -> str:
        summarized_settings = summarize_settings(settings)
        system_message = "You are a world-class hydrogeologist."
        prompt = f'''Based on the following CONFLUENCE model domain, generate a detailed and extensive perceptual model summary for the domain being modelled, 
                     citing the relevant literature and providing a list of references. Include key  processes and their interaction. 
                     Summarize previous modelling efforts in this basin and their findings. Identify modelling approaches that have provided good results or 
                     are likely to provide good results. Also identify (if available in the literature) modelling approaches that have not proven fruitful.:\n\n{summarized_settings}'''
        perceptual_model = self.api.generate_text(prompt, system_message)
        return perceptual_model

class MeteorologicalExpert(Expert):
    def __init__(self, api: AnvilGPTAPI):
        super().__init__("Meteorological Expert", "evaluation of hydrological model performance", api)
    
    def generate_perceptual_model(self, settings: Dict[str, Any]) -> str:
        summarized_settings = summarize_settings(settings)
        system_message = "You are a world-class meteorologist."
        prompt = f'''Based on the following CONFLUENCE model domain, generate a detailed and extensive perceptual model summary for the domain being modelled, 
                     citing the relevant literature and providing a list of references. Include key  processes and their interaction. 
                     Summarize previous modelling efforts in this basin and their findings. Identify modelling approaches that have provided good results or 
                     are likely to provide good results. Also identify (if available in the literature) modelling approaches that have not proven fruitful.:\n\n{summarized_settings}'''
        perceptual_model = self.api.generate_text(prompt, system_message)
        return perceptual_model

class Chairperson:
    def __init__(self, experts: List[Expert], api: AnvilGPTAPI):
        self.experts = experts
        self.api = api

    def load_control_file(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    def consult_experts(self, settings: Dict[str, Any], confluence_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        synthesis = {}
        for expert in self.experts:
            synthesis[expert.name] = expert.analyze_settings(settings, confluence_results)
        return synthesis

    def generate_report(self, settings: Dict[str, Any], synthesis: Dict[str, Any], confluence_results: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
        summarized_settings = summarize_settings(settings)
        
        system_message = "You are the chairperson of INDRA."
        prompt = f"Summarize the following expert analyses as a panel discussion:\n\n"
        for expert_name, analysis in synthesis.items():
            prompt += f"{expert_name} Analysis: {analysis['full_analysis']}\n\n"
        
        if confluence_results:
            prompt += f"CONFLUENCE Results: {confluence_results}\n\n"
        
        panel_summary = self.api.generate_text(prompt, system_message)
        
        prompt = f"Based on the panel discussion and settings, provide a concluded summary and suggest improvements:\n\nPanel Discussion: {panel_summary}\n\nSettings: {summarized_settings}"
        if confluence_results:
            prompt += f"\n\nCONFLUENCE Results: {confluence_results}"
        
        conclusion = self.api.generate_text(prompt, system_message)
        
        suggestions = self._extract_suggestions(conclusion)
        
        return {
            "concluded_summary": conclusion,
            "panel_summary": panel_summary
        }, suggestions

    def _extract_suggestions(self, conclusion: str) -> Dict[str, Any]:
        system_message = "You are the chairperson of INDRA."
        prompt = f"""Extract key suggestions for improving the CONFLUENCE model configuration from this conclusion. Format as a Python dictionary:

        Conclusion:
        {conclusion}

        Format your response as:
        SUGGESTIONS DICTIONARY:
        suggestions = {{
            "PARAMETER1": "suggested change 1",
            "PARAMETER2": "suggested change 2",
        }}

        SUMMARY:
        <Brief summary of suggestions and impact>
        """

        response = self.api.generate_text(prompt, system_message)
        
        dict_part = response.split("SUGGESTIONS DICTIONARY:")[1].split("SUMMARY:")[0].strip()
        local_vars = {}
        exec(dict_part, globals(), local_vars)
        suggestions = local_vars['suggestions']

        return suggestions

    def expert_initiation(self, watershed_name: str) -> Tuple[Dict[str, Any], str]:
        """Consult experts to determine optimal initial settings for the given watershed."""
        system_message = "You are the chairperson of INDRA."
        prompt = """
        We are initiating a new CONFLUENCE project for the watershed: {}

        Suggest optimal initial settings for:
        1. HYDROLOGICAL_MODEL (SUMMA, FLASH)
        2. ROUTING_MODEL (mizuroute)
        3. FORCING_DATASET (RDRS, ERA5)
        4. DOMAIN_DISCRETIZATION (elevation, soilclass, landclass)
        5. ELEVATION_BAND_SIZE
        6. MIN_HRU_SIZE
        7. POUR_POINT_COORDS (lat/lon)
        8. BOUNDING_BOX_COORDS (lat_max/lon_min/lat_min/lon_max.)

        Your response MUST follow this EXACT format (including the triple backticks):
        ```python
        config = {{
            "HYDROLOGICAL_MODEL": "SUMMA",  # or "FLASH"
            "ROUTING_MODEL": "mizuroute",
            "FORCING_DATASET": "ERA5",  # or "RDRS"
            "DOMAIN_DISCRETIZATION": "elevation",  # or "soilclass" or "landclass"
            "ELEVATION_BAND_SIZE": 100,  # example value
            "MIN_HRU_SIZE": 1,  # example value
            "POUR_POINT_COORDS": "60.0/-135.0",  # example coordinates
            "BOUNDING_BOX_COORDS": "62.0/58.0/-130.0/-140.0"  # example bounding box
        }}
        ```

        After the configuration block, provide a detailed justification for each choice.
        Include reasoning based on the watershed characteristics and available data.
        """.format(watershed_name)
        
        try:
            print("\nRequesting configuration from AnvilGPT...")
            response = self.api.generate_text(prompt, system_message)
            
            print("\nParsing response...")
            print(f"Response length: {len(response)}")
            
            # First, try to extract the code block
            if "```python" not in response:
                print("\nNo Python code block found. Looking for config dictionary directly...")
                # Try to find the config dictionary directly
                if "config = {" in response:
                    start_idx = response.find("config = {")
                    end_idx = response.find("}", start_idx) + 1
                    code_block = response[start_idx:end_idx]
                else:
                    raise ValueError("Could not find configuration dictionary in response")
            else:
                # Split by code block markers
                parts = response.split("```")
                if len(parts) < 3:
                    print("\nIncomplete code block found. Response parts:", len(parts))
                    raise ValueError("Incomplete code block in response")
                
                # Get the python code block (should be the second part)
                code_block = parts[1].replace('python', '').strip()
            
            print("\nExtracting configuration...")
            print(f"Code block found: {code_block[:100]}...")  # Print first 100 chars
            
            # Create a new dictionary to store the configuration
            local_vars = {}
            
            try:
                # Execute the code block in a safe context
                exec(code_block, {"__builtins__": {}}, local_vars)
            except Exception as e:
                print(f"\nError executing code block: {str(e)}")
                print("Code block content:")
                print(code_block)
                raise
            
            # Get the config dictionary
            if 'config' not in local_vars:
                raise ValueError("Configuration dictionary not found in response")
            
            config = local_vars['config']
            print("\nConfiguration extracted successfully.")
            
            # Extract justification (everything after the configuration)
            try:
                if "```" in response:
                    justification = response.split("```")[-1].strip()
                else:
                    end_idx = response.find("}", response.find("config = {")) + 1
                    justification = response[end_idx:].strip()
                
                if not justification:
                    justification = "No detailed justification provided in the response."
            except Exception as e:
                print(f"\nError extracting justification: {str(e)}")
                justification = "Error extracting justification from response."
            
            print("\nValidating configuration...")
            # Validate the required keys are present
            required_keys = {
                "HYDROLOGICAL_MODEL", "ROUTING_MODEL", "FORCING_DATASET", 
                "STREAM_THRESHOLD", "DOMAIN_DISCRETIZATION", "ELEVATION_BAND_SIZE", 
                "MIN_HRU_SIZE", "POUR_POINT_COORDS", "BOUNDING_BOX_COORDS"
            }
            
            missing_keys = required_keys - set(config.keys())
            if missing_keys:
                raise ValueError(f"Missing required configuration keys: {missing_keys}")
                
            # Validate the values
            if not isinstance(config["HYDROLOGICAL_MODEL"], str) or config["HYDROLOGICAL_MODEL"] not in ["SUMMA", "FLASH"]:
                raise ValueError("Invalid HYDROLOGICAL_MODEL value")
            if not isinstance(config["ROUTING_MODEL"], str) or config["ROUTING_MODEL"] != "mizuroute":
                raise ValueError("Invalid ROUTING_MODEL value")
            if not isinstance(config["FORCING_DATASET"], str) or config["FORCING_DATASET"] not in ["RDRS", "ERA5"]:
                raise ValueError("Invalid FORCING_DATASET value")
            
            print("\nConfiguration validation successful.")
            return config, justification
            
        except Exception as e:
            print(f"\nError processing configuration: {str(e)}")
            print("\nFalling back to default configuration...")
            
            # Provide a default configuration as fallback
            default_config = {
                "HYDROLOGICAL_MODEL": "SUMMA",
                "ROUTING_MODEL": "mizuroute",
                "FORCING_DATASET": "ERA5",
                "STREAM_THRESHOLD": 100,
                "DOMAIN_DISCRETIZATION": "elevation",
                "ELEVATION_BAND_SIZE": 100,
                "MIN_HRU_SIZE": 1,
                "POUR_POINT_COORDS": "60.0/-135.0",  # Default Yukon coordinates
                "BOUNDING_BOX_COORDS": "62.0/58.0/-130.0/-140.0"  # Default Yukon bounding box
            }
            
            default_justification = f"""
            Using default configuration values for {watershed_name} watershed. Please update the following parameters:
            - POUR_POINT_COORDS: Currently set to default Yukon coordinates
            - BOUNDING_BOX_COORDS: Currently set to default Yukon bounding box
            Other parameters can be adjusted based on specific watershed characteristics.
            
            Default configuration explanation:
            - HYDROLOGICAL_MODEL: SUMMA (comprehensive physics-based model)
            - ROUTING_MODEL: mizuroute (standard routing model)
            - FORCING_DATASET: ERA5 (globally available, good quality)
            - STREAM_THRESHOLD: 100 (moderate detail level)
            - DOMAIN_DISCRETIZATION: elevation (standard approach)
            - ELEVATION_BAND_SIZE: 100m (reasonable resolution)
            - MIN_HRU_SIZE: 1 km² (standard minimum size)
            """
            
            return default_config, default_justification

class INDRA:
    def __init__(self):
        bearer_token = os.environ.get('ANVIL_GPT_API_KEY')
        if not bearer_token:
            raise ValueError("ANVIL_GPT_API_KEY not found in environment variables")

        self.api = AnvilGPTAPI(bearer_token)
        self.experts = [
            HydrologistExpert(self.api),
            DataScienceExpert(self.api),
            HydrogeologyExpert(self.api),
            MeteorologicalExpert(self.api)
        ]
        self.chairperson = Chairperson(self.experts, self.api)

    def _generate_perceptual_models(self, watershed_name: str) -> Dict[str, str]:
        print("Consulting domain experts for perceptual model generation...")
        
        perceptual_models = {}
        domain_experts = [expert for expert in self.experts 
                        if isinstance(expert, (HydrologistExpert, HydrogeologyExpert, MeteorologicalExpert))]
        
        settings = {"DOMAIN_NAME": watershed_name}
        
        for expert in domain_experts:
            print(f"\nGenerating {expert.name} perceptual model...")
            perceptual_models[expert.name] = expert.generate_perceptual_model(settings)
        
        return perceptual_models

    def _save_perceptual_models(self, file_path: Path, perceptual_models: Dict[str, str]):
        with open(file_path, 'w') as f:
            f.write("INDRA Perceptual Models Report\n")
            f.write("=============================\n\n")
            
            for expert_name, model in perceptual_models.items():
                f.write(f"{expert_name} Perceptual Model\n")
                f.write("-" * (len(expert_name) + 17) + "\n")
                f.write(model)
                f.write("\n\n")

    def run(self, control_file_path: Optional[Path] = None, confluence_results: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
        is_new_project = control_file_path is None
        
        if is_new_project:
            print("Initiating a new CONFLUENCE project.")
            watershed_name = input("Enter the name of the watershed you want to model: ")
        else:
            settings = self.chairperson.load_control_file(control_file_path)
            watershed_name = settings.get('DOMAIN_NAME')

        print("\nGenerating perceptual models for the domain...")
        perceptual_models = self._generate_perceptual_models(watershed_name)
        
        report_path = Path(os.getcwd()) / "indra_reports"
        report_path.mkdir(parents=True, exist_ok=True)
        
        perceptual_model_file = report_path / f"perceptual_model_{watershed_name}.txt"
        self._save_perceptual_models(perceptual_model_file, perceptual_models)
        
        print(f"\nPerceptual models saved to: {perceptual_model_file}")
        
        if input("\nContinue with configuration? (y/n): ").lower() != 'y':
            return {}, {}

        if is_new_project:
            config, justification = self.chairperson.expert_initiation(watershed_name)
            
            rationale_file = report_path / f"initial_decision_rationale_{watershed_name}.txt"
            with open(rationale_file, 'w') as f:
                f.write(f"INDRA Initial Configuration Decisions for {watershed_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write("Configuration Parameters:\n")
                f.write("-" * 35 + "\n")
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
                f.write("\nJustification:\n")
                f.write("-" * 13 + "\n")
                f.write(justification)
            
            print(f"\nConfiguration rationale saved to: {rationale_file}")
            
            # Create configuration directory
            config_path = Path("0_config_files")
            config_path.mkdir(parents=True, exist_ok=True)
            config_file_path = config_path / f"config_{watershed_name}.yaml"
            
            # Use template to create new config file
            template_path = Path(__file__).parent / '0_config_files' / 'config_template.yaml'
            self._create_config_file_from_template(
                template_path=template_path,
                output_path=config_file_path,
                watershed_name=watershed_name,
                expert_config=config
            )
            
            # Create/update symlink with force flag
            active_config_path = config_path / "config_active.yaml"
            if active_config_path.exists():
                active_config_path.unlink()
            try:
                os.symlink(config_file_path, active_config_path)
            except FileExistsError:
                os.remove(active_config_path)
                os.symlink(config_file_path, active_config_path)
            
            settings = config
            control_file_path = config_file_path
            
            # Run CONFLUENCE with initial configuration
            print("\nRunning CONFLUENCE with initial configuration...")
            confluence_results = self.run_confluence(config_file_path)
        
        synthesis = self.chairperson.consult_experts(settings, confluence_results)
        report, suggestions = self.chairperson.generate_report(settings, synthesis, confluence_results)
        
        self._save_synthesis_report(report, suggestions, watershed_name, report_path)
        
        print("\nINDRA Analysis Summary:")
        print("------------------------")
        if not is_new_project:
            print(f"Analyzed config file: {control_file_path}")
        print("\nKey points from analysis:")
        for i, key_point in enumerate(report['concluded_summary'].split('\n')[:10], 1):
            print(f"{i}. {key_point}")
        
        print("\nSuggestions for improvement:")
        for param, suggestion in suggestions.items():
            print(f"{param}: {suggestion}")
        
        return report, suggestions

    def _modify_configuration(self, settings: Dict[str, Any], expert_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Allow user to modify INDRA-suggested configuration settings interactively."""
        updated_settings = settings.copy()
        print("\nINDRA-suggested configuration settings:")
        modifiable_settings = {k: v for k, v in updated_settings.items() if k in expert_config}
        
        for key, value in modifiable_settings.items():
            print(f"{key}: {value}")
        
        while True:
            print("\nEnter setting key to modify (or 'done' to finish, 'cancel' to discard changes):")
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
                    print(f"Warning: Could not convert to type {type(current_value).__name__}, storing as string")
                
                updated_settings[key] = new_value
                print(f"Updated {key} to: {new_value}")
            else:
                print(f"Setting '{key}' is not an INDRA-suggested configuration.")
                print("Modifiable settings are:", ', '.join(modifiable_settings.keys()))
        
        return updated_settings

    def _create_config_file_from_template(self, template_path: Path, output_path: Path, 
                                    watershed_name: str, expert_config: Dict[str, Any]):
        """Create a new configuration file from template while preserving structure and comments."""
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found at: {template_path}")
        
        with open(template_path, 'r') as f:
            template_lines = f.readlines()
        
        expert_config['DOMAIN_NAME'] = watershed_name
        
        with open(output_path, 'w') as f:
            for line in template_lines:
                if line.strip().startswith('#') or line.strip().startswith('### ==='):
                    f.write(line)
                    continue
                    
                if ':' in line:
                    key = line.split(':')[0].strip()
                    if key in expert_config:
                        comment = line.split('#')[1].strip() if '#' in line else ''
                        value = expert_config[key]
                        if isinstance(value, str):
                            value = f"'{value}'" if ' ' in value else value
                        new_line = f"{key}: {value}"
                        if comment:
                            new_line += f"  # {comment}"
                        f.write(new_line + '\n')
                    else:
                        f.write(line)
                else:
                    f.write(line)

    def _create_config_file_from_template(self, template_path: Path, output_path: Path, watershed_name: str, expert_config: Dict[str, Any]):
        """
        Create a new configuration file from template while preserving structure and comments.

        Args:
            template_path (Path): Path to the template config file
            output_path (Path): Path where to save the new config file
            watershed_name (str): Name of the watershed
            expert_config (Dict[str, Any]): Expert-suggested configurations
        """
        if not template_path.exists():
            raise FileNotFoundError(f"Configuration template not found at: {template_path}")
        
        # Read template file preserving all lines
        with open(template_path, 'r') as f:
            template_lines = f.readlines()
        
        # Add watershed name to expert config
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
                        # Extract any inline comments
                        comment = line.split('#')[1].strip() if '#' in line else ''
                        value = expert_config[key]
                        
                        # Handle string values with spaces
                        if isinstance(value, str) and ' ' in value:
                            value = f"'{value}'"
                        
                        # Construct new line
                        new_line = f"{key}: {value}"
                        if comment:
                            new_line += f"  # {comment}"
                        f.write(new_line + '\n')
                    else:
                        # Keep original line for non-expert configs
                        f.write(line)
                else:
                    f.write(line)

    def analyze_confluence_results(self, confluence_results: Dict[str, Any]) -> str:
        """Analyze the results from a CONFLUENCE run."""
        system_message = "You are an expert in analyzing hydrological model results."
        
        prompt = f"""
        Please analyze these CONFLUENCE model results:

        {confluence_results}

        Provide a brief summary of model performance, highlighting notable aspects or potential issues.
        """

        analysis = self.api.generate_text(prompt, system_message)
        return analysis

    def run_confluence(self, config_path: Path) -> Dict[str, Any]:
        """Run CONFLUENCE with the given configuration file."""
        print(config_path)
        try:
            confluence = CONFLUENCE(config_path)
            confluence.run_workflow()
            return confluence.get_results()
        except Exception as e:
            print(f"Error running CONFLUENCE: {str(e)}")
            return {"error": str(e)}

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

if __name__ == "__main__":
    try:
        indra = INDRA()
        
        use_existing = input("Do you want to use an existing config file? (y/n): ").lower() == 'y'
        
        if use_existing:
            while True:
                print("\nEnter the path to your configuration file:")
                print("(You can use absolute path or relative path from current directory)")
                control_file_input = input().strip()
                
                control_file_path = Path(control_file_input).resolve()
                
                if not control_file_path.exists():
                    print(f"\nError: File not found: {control_file_path}")
                    retry = input("Try another path? (y/n): ").lower() == 'y'
                    if not retry:
                        print("Exiting program.")
                        sys.exit()
                elif not control_file_path.suffix in ['.yaml', '.yml']:
                    print(f"\nError: File must be a YAML file (.yaml or .yml)")
                    retry = input("Try another path? (y/n): ").lower() == 'y'
                    if not retry:
                        print("Exiting program.")
                        sys.exit()
                else:
                    try:
                        with open(control_file_path, 'r') as f:
                            yaml.safe_load(f)
                        print(f"\nUsing configuration file: {control_file_path}")
                        break
                    except yaml.YAMLError:
                        print(f"\nError: Invalid YAML format in {control_file_path}")
                        retry = input("Try another path? (y/n): ").lower() == 'y'
                        if not retry:
                            print("Exiting program.")
                            sys.exit()
        else:
            control_file_path = None
        
        confluence_results = None
        indra.run(control_file_path, confluence_results)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo set up your API key:")
        print("\nFor Unix-like systems (Linux/Mac):")
        print("1. Add to ~/.bashrc or ~/.zshrc:")
        print('   export ANVIL_GPT_API_KEY="your-api-key-here"')
        print("2. Run: source ~/.bashrc (or source ~/.zshrc)")
        print("\nFor Windows:")
        print("1. Open System Properties -> Advanced -> Environment Variables")
        print("2. Add new User Variable:")
        print("   Name: ANVIL_GPT_API_KEY")
        print("   Value: your-api-key")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Error details: {str(e)}")