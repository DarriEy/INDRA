import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests # type: ignore
import json
import os
import sys


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
        
        # Combine system message and prompt
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
            
            # Handle streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        if 'message' in json_response:
                            content = json_response['message'].get('content', '')
                            full_response += content
                            # Optional: Print content as it arrives for debugging
                            print(content, end='', flush=True)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        print(f"Problematic line: {line}")
                        continue
                        
            return full_response.strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"AnvilGPT API error: {str(e)}")

class Expert:
    """Base class for expert agents."""
    def __init__(self, name: str, expertise: str, api: AnvilGPTAPI):
        self.name = name
        self.expertise = expertise
        self.api = api
        self.prompt = EXPERT_PROMPTS[name]  # We'll define this at the top of the file

    def analyze_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Basic analysis method that can be overridden by specific experts."""
        summarized_settings = summarize_settings(settings)
        system_message = f"You are a world-class expert in {self.expertise}."
        prompt = f"{self.prompt}\n\nAnalyze these settings:\n\n{summarized_settings}"
        
        analysis = self.api.generate_text(prompt, system_message)
        return {"full_analysis": analysis}

def summarize_settings(settings: Dict[str, Any], max_length: int = 2000) -> str:
    """Utility function to summarize settings."""
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

# Test function
def test_anvil_gpt():
    """Test the Anvil GPT API implementation."""
    try:
        # Get bearer token from environment
        bearer_token = os.environ.get('ANVIL_GPT_API_KEY')
        if not bearer_token:
            raise ValueError("ANVIL_GPT_API_KEY not found in environment variables")
        
        # Initialize API
        api = AnvilGPTAPI(bearer_token)
        
        # Test simple prompt
        print("Testing simple prompt...")
        response = api.generate_text(
            prompt="Explain what a hydrological model is in one sentence.",
            system_message="You are an expert in hydrology."
        )
        print("\nResponse:", response)
        
        # Test longer analysis
        print("\nTesting longer analysis...")
        test_settings = {
            "HYDROLOGICAL_MODEL": "SUMMA",
            "DOMAIN_NAME": "TestBasin",
            "FORCING_DATASET": "ERA5"
        }
        
        expert = Expert("Test Expert", "hydrology", api)
        analysis = expert.analyze_settings(test_settings)
        print("\nAnalysis:", analysis['full_analysis'])
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        if isinstance(e, ValueError) and "ANVIL_GPT_TOKEN" in str(e):
            print("\nTo set up your Anvil GPT token:")
            print("Export ANVIL_GPT_TOKEN='your-bearer-token' in your environment")

if __name__ == "__main__":
    test_anvil_gpt()