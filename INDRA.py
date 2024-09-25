import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any, List
import anthropic # type: ignore
import os

class AnthropicAPI:
    """A wrapper for the Anthropic API."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
    """Base class for all experts in the INDRA system."""

    def __init__(self, name: str, expertise: str, api: AnthropicAPI):
        self.name = name
        self.expertise = expertise
        self.api = api

    def analyze_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the given settings and provide insights."""
        summarized_settings = summarize_settings(settings)
        system_message = f"You are a world-class expert in {self.expertise} with extensive knowledge of the CONFLUENCE model. Provide insightful analysis of the given model settings, focusing on your area of expertise."
        prompt = f"Analyze the following CONFLUENCE model settings, focusing on {self.expertise}. Provide insights and suggestions:\n\n{summarized_settings}"
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
        prompt = f"Based on the following CONFLUENCE model settings, generate a perceptual model summary for the domain being modelled. Include key hydrological processes and their interactions:\n\n{summarized_settings}"
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

    def consult_experts(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Consult all experts and gather their analyses."""
        synthesis = {}
        for expert in self.experts:
            synthesis[expert.name] = expert.analyze_settings(settings)
        return synthesis

    def generate_report(self, settings: Dict[str, Any], synthesis: Dict[str, Any]) -> Dict[str, str]:
        """Generate a comprehensive report based on expert analyses."""
        summarized_settings = summarize_settings(settings)
        
        # Generate panel discussion summary
        system_message = "You are the chairperson of INDRA. Summarize the expert panel discussion based on their analyses."
        prompt = f"Summarize the following expert analyses of CONFLUENCE model settings as if it were a panel discussion:\n\n"
        for expert_name, analysis in synthesis.items():
            prompt += f"{expert_name} Analysis: {analysis['full_analysis']}\n\n"
        panel_summary = self.api.generate_text(prompt, system_message, max_tokens=1500)
        
        # Generate concluded summary
        system_message = "You are the chairperson of INDRA. Provide a concluded summary of the expert analyses."
        prompt = f"Based on the following expert analyses and panel discussion, provide a concluded summary for the CONFLUENCE model settings:\n\nPanel Discussion: {panel_summary}\n\nSettings: {summarized_settings}"
        concluded_summary = self.api.generate_text(prompt, system_message, max_tokens=1000)
        
        return {
            "concluded_summary": concluded_summary,
            "panel_summary": panel_summary
        }

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
        self.api = AnthropicAPI()
        self.experts = [
            HydrologistExpert(self.api),
            DataPreprocessingExpert(self.api),
            ModelCalibrationExpert(self.api),
            GeospatialAnalysisExpert(self.api),
            PerformanceMetricsExpert(self.api)
        ]
        self.chairperson = Chairperson(self.experts, self.api)

    def run(self, control_file_path: Path) -> None:
        """
        Run the INDRA analysis on the given CONFLUENCE control file.

        Args:
            control_file_path (Path): Path to the CONFLUENCE control file.

        Returns:
            None: The method generates and saves a report file.
        """
        settings = self.chairperson.load_control_file(control_file_path)
        synthesis = self.chairperson.consult_experts(settings)
        report = self.chairperson.generate_report(settings, synthesis)
        
        # Generate perceptual model summary
        hydrologist = next(expert for expert in self.experts if isinstance(expert, HydrologistExpert))
        perceptual_model = hydrologist.generate_perceptual_model(settings)
        
        # Save the report
        report_path = control_file_path.parent / f"{control_file_path.stem}_INDRA_report.txt"
        with open(report_path, 'w') as f:
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
        
        print(f"INDRA report saved to: {report_path}")

        return report, synthesis

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
    indra = INDRA()
    control_file_path = Path("/Users/darrieythorsson/compHydro/code/CONFLUENCE/0_config_files/config_active.yaml")
    report, synthesis = indra.run(control_file_path)
    print(report)
    print(synthesis)