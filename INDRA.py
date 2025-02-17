import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import anthropic # type: ignore
import os
import sys
import subprocess
import time
import shutil

sys.path.append(str(Path(__file__).resolve().parent.parent))
from CONFLUENCE.CONFLUENCE import CONFLUENCE # type: ignore

CONFLUENCE_OVERVIEW = """
CONFLUENCE (Community Optimization and Numerical Framework for Large-domain Understanding of Environmental Networks and Computational Exploration) is an integrated hydrological modeling platform. It combines various components for data management, model setup, optimization, uncertainty analysis, forecasting, and visualization across multiple scales and regions.

Key features of CONFLUENCE include:
1. Support for multiple hydrological models (currently SUMMA, FLASH, FUSE, GR, HYPE, MESH)
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
    """,

    "Geographer Expert": f"""
    {CONFLUENCE_OVERVIEW}
    
    As the Geographer Expert, your role is to determine precise spatial boundaries and coordinates for watershed delineation across broad geographic regions. Consider the following aspects:
    1. Regional Drainage Systems
        - Identify major river systems and their interconnections
        - Consider regional topography and drainage patterns
        - Account for potential underground karst systems or complex drainage networks
        
    2. Watershed Extent Determination
        - Always extend boundaries significantly beyond expected watershed limits
        - Consider regional geological features that might influence drainage
        - Include all potential tributary headwaters in the broader region
        - Account for seasonal variations in watershed extent
        
    3. Coordinate Selection Principles
        For pour points:
        - Must be on the main river channel
        - Must be at least 10km upstream from any major confluence
        - Must be at least 15km upstream from any estuary
        - Verify against known stream networks
        - Specify with 4 decimal places precision
        
        For bounding boxes:
        - Must extend at least 50km beyond ANY potential watershed boundary
        - Must include entire headwater regions of ALL potential tributaries
        - Must account for regional topographic features that could influence drainage
        - Must include substantial margin beyond the furthest potential stream origins
        - Must fully encompass any regional drainage divides
        - Must account for uncertainty in watershed boundaries
        - Specify with 2 decimal places precision
        
    4. Validation Requirements
        - Pour point must be well within bounds (at least 30km from box edges)
        - Bounding box must include complete regional drainage systems
        - Must account for both surface and potential subsurface drainage patterns
        
    When specifying coordinates:
    - Pour points: Always select a point that is unambiguously on the main stem
    - Bounding box: Always err on the side of including too much rather than too little area
    - If in doubt about extent, add an additional 25km buffer to all sides
    
    Your goal is to ensure the selected region is so expansive that it is impossible to miss any part of the watershed or its contributing areas. It's far better to include too much area than risk missing any part of the drainage system.
    """,
}


class AnthropicAPI:
    """
    Wrapper for Anthropic's Claude API providing controlled access to language model capabilities.
    
    Manages interactions with the Anthropic API, handling prompt construction,
    response processing, and error handling for expert consultations.

    Attributes:
        client (anthropic.Anthropic): Authenticated Anthropic API client
    """

    def __init__(self, api_key):
        """
        Initialize Anthropic API client with authentication.

        Args:
            api_key (str): Anthropic API authentication key
        """
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_text(self, prompt: str, system_message: str, max_tokens: int = 1750) -> str:
        """
        Generate text response using Anthropic's Claude model.

        Creates a structured prompt combining system context and user query,
        manages token limits, and processes model response.

        Args:
            prompt (str): Main prompt/query text
            system_message (str): System context/instruction message
            max_tokens (int, optional): Maximum response length. Defaults to 1750.

        Returns:
            str: Generated response text

        Raises:
            anthropic.APIError: If API request fails
        """
        message = self.client.messages.create(
            model="claude-3-5-sonnet-latest",
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
    """
    Base class for specialized AI expert agents in the INDRA system.
    
    Each expert agent provides domain-specific analysis of CONFLUENCE model configurations
    and watershed characteristics. Experts utilize the Anthropic API to generate insights
    within their area of expertise.

    Attributes:
        name (str): Identifier for the expert
        expertise (str): Domain of expertise
        api (AnthropicAPI): Interface to language model API
        prompt (str): Expert-specific prompt template for analysis
    """
    def __init__(self, name: str, expertise: str, api: AnthropicAPI):
        self.name = name
        self.expertise = expertise
        self.api = api
        self.prompt = EXPERT_PROMPTS[name]

    def analyze_settings(self, settings: Dict[str, Any], confluence_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze CONFLUENCE model settings from expert's domain perspective.

        Generates domain-specific insights and recommendations based on model configuration
        and results if available.

        Args:
            settings (Dict[str, Any]): CONFLUENCE model configuration settings
            confluence_results (Optional[Dict[str, Any]]): Results from model execution

        Returns:
            Dict[str, Any]: Analysis results containing:
                - full_analysis (str): Detailed expert analysis and recommendations
        """
        summarized_settings = summarize_settings(settings)
        system_message = f"You are a world-class expert in {self.expertise} with extensive knowledge of the CONFLUENCE model. Provide insightful analysis of the given model settings, focusing on your area of expertise."
        prompt = f"{self.prompt}\n\nAnalyze the following CONFLUENCE model settings, focusing on {self.expertise}. Provide insights and suggestions:\n\n{summarized_settings}"
        
        if confluence_results:
            prompt += f"\n\nCONFLUENCE Results: {confluence_results}"
        
        analysis = self.api.generate_text(prompt, system_message)
        return {"full_analysis": analysis}



class HydrologistExpert(Expert):
    """
    Expert agent specializing in hydrological processes and model structure.
    
    Provides analysis focusing on:
    - Appropriateness of model structure for watershed
    - Representation of hydrological processes
    - Process parameterization
    - Scale considerations
    - Dominant hydrological processes
    """

    def __init__(self, api: AnthropicAPI):
        super().__init__("Hydrologist Expert", "hydrological processes and model structure", api)

    def generate_perceptual_model(self, settings: Dict[str, Any]) -> str:
        """
        Generate hydrological perceptual model for watershed.

        Creates detailed conceptual model of watershed hydrological behavior,
        incorporating:
        - Key hydrological processes
        - Process interactions and connectivity
        - Spatial and temporal dynamics
        - Previous modeling insights from literature

        Args:
            settings (Dict[str, Any]): Basic watershed configuration settings

        Returns:
            str: Detailed perceptual model description with literature references
        """
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

class GeographerExpert(Expert):
    """Expert in watershed delineation and spatial boundary determination."""
    
    def __init__(self, api: AnthropicAPI):
        super().__init__("Geographer Expert", "watershed delineation and spatial boundaries", api)
        
    def validate_coordinates(self, pour_point: str, bounding_box: str) -> Dict[str, Any]:
        """
        Validate proposed coordinates against spatial criteria.
        """
        system_message = """You are a world-class expert geographer validating watershed coordinates.
        You must respond in the exact format specified, with each section clearly marked and separated by double newlines."""
        
        prompt = f"""
        Analyze and validate these watershed coordinates:
        Pour point: {pour_point}
        Bounding box: {bounding_box}

        Requirements:
        1. Pour point must:
           - Be on a main river channel
           - Be at least 10km upstream from confluences
           - Be at least 15km upstream from estuaries
           - Use 6 decimal places
        
        2. Bounding box must:
           - Extend 50km beyond watershed boundaries
           - Include all tributary headwaters
           - Have significant margin beyond divides
           - Fully contain pour point with 30km margin
           - Use 2 decimal places

        YOU MUST STRUCTURE YOUR RESPONSE EXACTLY AS FOLLOWS, WITH EACH SECTION SEPARATED BY DOUBLE NEWLINES:

        VALIDATION:
        [Write only YES or NO]

        POUR_POINT:
        [If changes needed, write new coordinates in exact format lat/lon. If no changes, write UNCHANGED]

        BOUNDING_BOX:
        [If changes needed, write new coordinates in exact format lat_max/lon_min/lat_min/lon_max. If no changes, write UNCHANGED]

        ADJUSTMENTS:
        [List each specific change made, one per line with leading hyphen]

        JUSTIFICATION:
        [Explain validation and changes]

        Example correct response format:
        VALIDATION:
        NO

        POUR_POINT:
        48.123456/2.123456

        BOUNDING_BOX:
        49.12/1.23/47.45/3.67

        ADJUSTMENTS:
        - Moved pour point 2km upstream to avoid confluence
        - Expanded northern boundary by 15km for safety margin
        - Expanded eastern boundary to include potential tributaries

        JUSTIFICATION:
        Original pour point was too close to confluence. Bounding box needed expansion to ensure complete watershed capture.
        """
        
        response = self.api.generate_text(prompt, system_message)
        
        try:
            # Split into sections
            sections = {
                section.split(':\n', 1)[0]: section.split(':\n', 1)[1].strip()
                for section in response.split('\n\n')
                if ':' in section
            }
            
            # Initialize result
            result = {
                "valid": sections.get('VALIDATION', '').strip().upper() == 'YES',
                "pour_point": pour_point,  # Default to original
                "bounding_box": bounding_box,  # Default to original
                "adjustments": [],
                "justification": sections.get('JUSTIFICATION', 'No justification provided')
            }
            
            # Process pour point
            new_pour_point = sections.get('POUR_POINT', '').strip()
            if new_pour_point and new_pour_point != 'UNCHANGED':
                if self._validate_pour_point_format(new_pour_point):
                    result['pour_point'] = new_pour_point
                    result['adjustments'].append(f"Updated pour point to {new_pour_point}")
                else:
                    raise ValueError(f"Invalid pour point format: {new_pour_point}")
            
            # Process bounding box
            new_bbox = sections.get('BOUNDING_BOX', '').strip()
            if new_bbox and new_bbox != 'UNCHANGED':
                if self._validate_bbox_format(new_bbox):
                    result['bounding_box'] = new_bbox
                    result['adjustments'].append(f"Updated bounding box to {new_bbox}")
                else:
                    raise ValueError(f"Invalid bounding box format: {new_bbox}")
            
            # Add adjustments
            adjustments_text = sections.get('ADJUSTMENTS', '')
            if adjustments_text:
                additional_adjustments = [
                    adj.strip('- ').strip()
                    for adj in adjustments_text.split('\n')
                    if adj.strip('- ').strip()
                ]
                result['adjustments'].extend(additional_adjustments)
            
            return result
            
        except Exception as e:
            print(f"Error parsing geographer response: {str(e)}")
            print(f"Raw response:\n{response}")
            
            return {
                "valid": False,
                "pour_point": pour_point,
                "bounding_box": bounding_box,
                "adjustments": ["Validation failed: formatting error in expert response"],
                "justification": f"Coordinate validation failed due to response parsing error: {str(e)}"
            }

    def _validate_pour_point_format(self, pour_point: str) -> bool:
        """Validate pour point coordinate format."""
        try:
            lat, lon = pour_point.split('/')
            lat_float, lon_float = float(lat), float(lon)
            # Check for reasonable coordinate ranges
            if not (-90 <= lat_float <= 90 and -180 <= lon_float <= 180):
                return False
            # Check decimal places
            if len(lat.split('.')[-1]) != 6 or len(lon.split('.')[-1]) != 6:
                return False
            return True
        except:
            return False

    def _validate_bbox_format(self, bbox: str) -> bool:
        """Validate bounding box coordinate format."""
        try:
            lat_max, lon_min, lat_min, lon_max = bbox.split('/')
            coords = [float(lat_max), float(lon_min), float(lat_min), float(lon_max)]
            # Check for reasonable coordinate ranges
            if not all(-90 <= c <= 90 for c in [coords[0], coords[2]]):
                return False
            if not all(-180 <= c <= 180 for c in [coords[1], coords[3]]):
                return False
            # Check decimal places
            if not all(len(str(c).split('.')[-1]) == 2 for c in coords):
                return False
            # Check logical ordering
            if coords[0] <= coords[2] or coords[3] <= coords[1]:
                return False
            return True
        except:
            return False

class Chairperson:
    """
    Coordinator for the INDRA expert panel system.
    
    The Chairperson manages expert consultation, synthesizes expert analyses,
    and generates comprehensive reports and recommendations. Acts as the primary
    interface between the expert panel and the INDRA system.

    Attributes:
        experts (List[Expert]): Panel of expert agents
        api (AnthropicAPI): Interface to language model API
    """
    def __init__(self, experts: List[Expert], api: AnthropicAPI):
        self.experts = experts
        self.api = api

    def load_control_file(self, file_path: Path) -> Dict[str, Any]:
        """Load the CONFLUENCE control file."""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    def consult_experts(self, settings: Dict[str, Any], confluence_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Coordinate expert panel consultation process.

        Gathers analyses from all experts and compiles their insights.

        Args:
            settings (Dict[str, Any]): CONFLUENCE configuration settings
            confluence_results (Optional[Dict[str, Any]]): Model execution results

        Returns:
            Dict[str, Any]: Compiled expert analyses keyed by expert name
        """
        synthesis = {}
        for expert in self.experts:
            synthesis[expert.name] = expert.analyze_settings(settings, confluence_results)
        return synthesis

    def generate_report(self, settings: Dict[str, Any], synthesis: Dict[str, Any], confluence_results: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Generate comprehensive analysis report from expert consultations.

        Synthesizes expert analyses into coherent report and actionable suggestions.

        Args:
            settings (Dict[str, Any]): Model configuration settings
            synthesis (Dict[str, Any]): Compiled expert analyses
            confluence_results (Optional[Dict[str, Any]]): Model execution results

        Returns:
            Tuple[Dict[str, str], Dict[str, Any]]: Contains:
                - Report dictionary with panel_summary and concluded_summary
                - Configuration suggestions dictionary
        """
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
        """
        Generate expert-guided initial configuration with validated coordinates.
        
        Args:
            watershed_name (str): Name of watershed to be modeled

        Returns:
            Tuple[Dict[str, Any], str]: Contains:
                - Initial configuration dictionary
                - Justification for configuration choices
        """
        # First get initial configuration from experts
        config, justification = self._get_initial_config(watershed_name)
        
        # Find the geographer expert to validate coordinates
        geographer = next((expert for expert in self.experts if isinstance(expert, GeographerExpert)), None)
        
        if geographer:
            try:
                # Validate coordinates through geographer expert
                validation = geographer.validate_coordinates(
                    config.get('POUR_POINT_COORDS'),
                    config.get('BOUNDING_BOX_COORDS')
                )
                
                if not validation['valid']:
                    # Update coordinates with validated versions
                    config['POUR_POINT_COORDS'] = validation['pour_point']
                    config['BOUNDING_BOX_COORDS'] = validation['bounding_box']
                    
                    # Add validation explanation to justification
                    justification += "\n\nCoordinate Adjustments:\n" + validation['justification']
                    
                    print("Coordinates adjusted based on geographer validation:")
                    for adj in validation['adjustments']:
                        print(f"- {adj}")
            except Exception as e:
                print(f"Error during coordinate validation: {str(e)}")
                raise
        
        return config, justification

    def _get_initial_config(self, watershed_name: str) -> Tuple[Dict[str, Any], str]:
        """Helper method to get initial configuration from expert panel."""
        system_message = '''You are the chairperson of INDRA, coordinating a panel of hydrological modeling experts 
                            to determine optimal initial settings for a CONFLUENCE model configuration.'''
        
        prompt = f"""
        We are initiating a new CONFLUENCE project for the watershed named: {watershed_name}

        As the panel of experts, please suggest optimal initial settings for the following configuration parameters:

        1. HYDROLOGICAL_MODEL (options: SUMMA, FUSE, FLASH)
        2. ROUTING_MODEL (options: mizuroute)
        3. FORCING_DATASET (options: RDRS, ERA5)
        4. DOMAIN_DISCRETIZATION method (options, elevation, soilclass, landclass)
        5. ELEVATION_BAND_SIZE (if using elevation-based discretization)
        6. MIN_HRU_SIZE: Minimum size of the model domain HRUs, in km2 recommended 10 km2 for large watersheds and 1 km2 for small watersheds
        7. POUR_POINT_COORDS: coordinates lat/lon to define watershed to delineate must be specified as decimals with 4 digits 
                            in the format 'lat/lon'. Select coordinates on the river main step, following the geographer's guidelines:
                            - Must be at least 10km upstream from any major confluence
                            - Must be at least 15km upstream from any estuary
                            - Must be verified against known stream networks
        8. BOUNDING_BOX_COORDS: coordinates of the bounding box must be specified as decimals with 2 digits 
                            in the format 'lat_max/lon_min/lat_min/lon_max'. Follow the geographer's guidelines:
                            - Must extend at least 50km beyond ANY potential watershed boundary
                            - Must include entire headwater regions of ALL potential tributaries
                            - Must account for regional topographic features
                            - Must include substantial margin beyond the furthest potential stream origins
        9. PARAMS_TO_CALIBRATE: If HYDROLOGICAL_MODEL is SUMMA, select which parameters to calibrate.

        For each parameter, provide a brief justification for your recommendation.

        IMPORTANT: 
        - Use proper Python syntax: True/False for booleans (not true/false)
        - For string values that contain spaces, enclose them in quotes
        - Be extremely generous with bounding box coordinates following geographer expert guidelines

        Present your response in the following format:

        CONFIG DICTIONARY:
        config = {{
            "HYDROLOGICAL_MODEL": "SUMMA",  # Example
            "DELINEATE_BY_POURPOINT": True,  # Example of proper boolean format
            ...
        }}

        JUSTIFICATION SUMMARY:
        <A concise summary of the justifications for the chosen settings>
        """
        
        response = self.api.generate_text(prompt, system_message, max_tokens=1500)
        
        # Split and process response
        config_part, justification_part = response.split("JUSTIFICATION SUMMARY:")
        config_code = config_part.split("CONFIG DICTIONARY:")[1].strip()
        config_code = config_code.replace("```python", "").replace("```", "").strip()
        
        # Clean and validate config code
        def clean_config_code(code: str) -> str:
            lines = code.split('\n')
            cleaned_lines = []
            
            for line in lines:
                if ':' in line:
                    key_part, value_part = line.split(':', 1)
                    if '#' in value_part:
                        value_part, comment = value_part.split('#', 1)
                        comment = f"#{comment}"
                    else:
                        comment = ""
                    
                    value_part = value_part.strip().strip(',')
                    
                    if value_part.lower() in ['true', 'false']:
                        value_part = value_part.title()
                    
                    line = f"{key_part}: {value_part}{', ' if ',' in line else ''}{comment}"
                
                cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines)
        
        cleaned_config_code = clean_config_code(config_code)
        
        try:
            local_vars = {}
            exec(cleaned_config_code, globals(), local_vars)
            config = local_vars['config']
            
            # Validate boolean values
            for key, value in config.items():
                if isinstance(value, str):
                    if value.lower() == 'true':
                        config[key] = True
                    elif value.lower() == 'false':
                        config[key] = False
        except Exception as e:
            print(f"Error processing configuration: {str(e)}")
            print(f"Problematic config code:\n{cleaned_config_code}")
            raise
        
        return config, justification_part.strip()

class INDRA:
    """
    Intelligent Network for Dynamic River Analysis (INDRA)

    INDRA is an AI-powered expert system for hydrological modeling analysis and configuration.
    It uses a panel of specialized AI experts to analyze and optimize CONFLUENCE model setups,
    generate perceptual models, and provide comprehensive insights for model improvement.

    The system coordinates multiple expert agents specializing in different aspects of 
    hydrological modeling (hydrology, data science, hydrogeology, meteorology) and synthesizes
    their insights through a chairperson agent.

    Key Features:
        - Perceptual model generation for watersheds
        - Automated initial configuration based on expert knowledge
        - Multi-expert analysis of model settings
        - Batch processing support for HPC environments
        - Comprehensive reporting and suggestions

    Attributes:
        api (AnthropicAPI): Interface to the Anthropic language model API
        experts (List[Expert]): Panel of expert AI agents
        chairperson (Chairperson): Coordinator for expert panel discussions

    Example:
        >>> indra = INDRA()
        >>> analysis_results, suggestions = indra.run("my_watershed_config.yaml")
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
            MeteorologicalExpert(self.api),
            GeographerExpert(self.api)  # Add the new expert
        ]
        self.chairperson = Chairperson(self.experts, self.api)

    def _generate_perceptual_models(self, watershed_name: str) -> Dict[str, str]:
        """
        Generate domain-specific perceptual models from multiple expert perspectives.
        
        Consults hydrologist, hydrogeologist, and meteorological experts to create
        comprehensive perceptual models of the watershed's behavior. Each expert 
        provides insights from their domain of expertise.

        Args:
            watershed_name (str): Name of the watershed to model

        Returns:
            Dict[str, str]: Dictionary mapping expert names to their perceptual model descriptions

        Note:
            The generated models combine theoretical knowledge with literature-based insights
            specific to the given watershed or similar watersheds.
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
        Save generated perceptual models to formatted text file.

        Args:
            file_path (Path): Output file path
            perceptual_models (Dict[str, str]): Expert perceptual models
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
        Save synthesis report and configuration suggestions.

        Args:
            report (Dict[str, str]): Analysis report
            suggestions (Dict[str, Any]): Configuration suggestions
            watershed_name (str): Name of watershed
            report_path (Path): Output directory path
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
            watershed_name = _sanitize_watershed_name(watershed_name)

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

            if shutil.which('sbatch') is not None:
                job_info = self.run_confluence_batch(control_file_path)
            else:
                job_info = self.run_confluence_interactive(control_file_path)
            
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
                elif status.startswith("UNKNOWN") or status.startswith("RUNNING"):
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

            pass

    def _check_job_status(self, job_id: str) -> str:
        """
        Check status of SLURM job.

        Args:
            job_id (str): SLURM job identifier

        Returns:
            str: Job status (PENDING, RUNNING, COMPLETED, FAILED, etc.)
        """
        try:
            cmd = f"squeue -j {job_id} --format=State --noheader --parsable2"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                # Get the last status (most recent) and strip whitespace
                status = result.stdout.strip().split('\n')[0]
                return status
            return "UNKNOWN"
        except Exception as e:
            print(f"Error checking job status: {str(e)}")
            return "UNKNOWN"

    def _read_confluence_results(self, config_path: Path, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Read and parse results from completed CONFLUENCE run.

        Args:
            config_path (Path): Configuration file path
            job_id (str): SLURM job identifier

        Returns:
            Optional[Dict[str, Any]]: Results dictionary if available
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
                print(f"CONFLUENCE output directory not found: {output_dir}")
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
            print(f"Error reading CONFLUENCE results: {str(e)}")
            return None
    
    def _modify_configuration(self, settings: Dict[str, Any], expert_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Interactively modify INDRA-suggested configuration settings.

        Args:
            settings (Dict[str, Any]): Current configuration settings
            expert_config (Dict[str, Any]): Expert-suggested configurations

        Returns:
            Optional[Dict[str, Any]]: Modified settings or None if cancelled
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
                        new_value = new_value in ('True', 'yes', '1', 'on')
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
        Generate configuration file from template with expert suggestions.

        Args:
            template_path (Path): Template configuration file path
            output_path (Path): Output configuration file path
            watershed_name (str): Name of watershed
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
    
    def run_confluence_interactive(self, config_path: Path) -> Dict[str, Any]:
        """
        Run CONFLUENCE with the given configuration file.

        Args:
            config_path (Path): Path to the configuration file.

        Returns:
            Dict[str, Any]: Results from the CONFLUENCE run.
        """
        print('Running CONFLUENCE in interactive mode')
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

    def run_confluence_batch(self, config_path: Path) -> Dict[str, Any]:
        """
        Execute CONFLUENCE model using HPC batch system (SLURM).

        Submits CONFLUENCE job to HPC queue and provides job monitoring capabilities.
        Creates necessary batch scripts and handles job submission process.

        Args:
            config_path (Path): Path to CONFLUENCE configuration file

        Returns:
            Dict[str, Any]: Job information including:
                - job_id: SLURM job identifier
                - status: Current job status
                - error: Error message if submission failed

        Raises:
            subprocess.CalledProcessError: If job submission fails
            FileNotFoundError: If configuration file not found
        """
        print('Running CONFLUENCE in batch mode')
        try:
            # Read the configuration file to get necessary parameters
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Create SLURM submission script
            slurm_script = self._create_slurm_script(config_path, config)
            
            # Submit job
            submit_cmd = f"sbatch run_confluence_batch.sh"
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
        Create SLURM batch submission script for CONFLUENCE execution.

        Args:
            config_path (Path): Path to configuration file
            config (Dict[str, Any]): Configuration parameters

        Returns:
            Path: Path to generated SLURM script
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
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=20G
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
    """
    Create concise summary of configuration settings within length limit.

    Args:
        settings (Dict[str, Any]): Configuration settings to summarize
        max_length (int): Maximum summary length in characters

    Returns:
        str: Truncated settings summary
    """
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

def _sanitize_watershed_name(name: str) -> str:
    """
    Clean watershed name for use in file paths and identifiers.

    Args:
        name (str): Raw watershed name

    Returns:
        str: Sanitized name with spaces replaced and special characters removed
    """
    # Replace spaces with underscores
    sanitized = name.strip().replace(' ', '_')
    # Remove any special characters that might cause issues in file names
    sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
    return sanitized

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