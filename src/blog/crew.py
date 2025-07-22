from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, after_kickoff
from crewai.tasks.task_output import TaskOutput
from typing import List, Set
import os
import re
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@CrewBase
class Blog():
    """Blog crew"""

    agents: List[Agent]
    tasks: List[Task]
    models_used: Set[str] # <--- NEW: Attribute to store unique model names

    def __init__(self):
        super().__init__()
        os.environ.setdefault("CREWAI_LOG_LEVEL", "DEBUG")
        self.models_used = set() # <--- NEW: Initialize the set for this crew run

    @staticmethod
    def _extract_model_from_response(raw_output: str):
        """
        (Internal Tool)
        Extracts the model name from the raw LLM response string.
        """
        try:
            # First, try to parse the whole string as JSON
            try:
                data = json.loads(raw_output)
                if 'model' in data:
                    return data['model']
            except (json.JSONDecodeError, TypeError):
                # If it fails, proceed to regex matching on the raw string
                pass

            # Regex for "model": "model_name"
            match = re.search(r'"model":\s*"([^"]+)"', raw_output)
            if match:
                return match.group(1)
        except Exception:
            # Ignore errors, as not all raw_output will contain model info
            pass
        
        return "Unknown"

    def _log_model_usage_callback(self, output: TaskOutput):
        """
        (NEW Callback Function)
        This function is called after each task is completed.
        It inspects the task's raw output to find the model name.
        """
        logger.info(f"Task completed. Analyzing output from agent: {output.agent}")
        
        # The raw response from the LLM is in `output.raw_output`
        raw_llm_response = output.raw_output
        
        model_name = self._extract_model_from_response(raw_llm_response)
        
        if model_name and model_name != "Unknown":
            logger.info(f"Extracted model from response: {model_name}")
            self.models_used.add(model_name)
        else:
            logger.warning("Could not determine model from raw output.")

    @after_kickoff
    def display_models_used(self):
        """
        (NEW Summary Function)
        This function runs after the entire crew has finished its work.
        """
        print("\n\n" + "="*60)
        print("🤖 CREW EXECUTION SUMMARY")
        print("="*60)
        if self.models_used:
            print("The following models were used during this run:")
            for model in self.models_used:
                print(f"  • {model}")
        else:
            print("No specific model names could be extracted during the run.")
        print("="*60)

    @agent
    def planner(self) -> Agent:
        return Agent(config=self.agents_config['planner'], verbose=True)

    @agent
    def writer(self) -> Agent:
        return Agent(config=self.agents_config['writer'], verbose=True)
    
    @agent
    def editor(self) -> Agent:
        return Agent(config=self.agents_config['editor'], verbose=True)

    @task
    def plan(self) -> Task:
        return Task(
            config=self.tasks_config['plan'],
            callback=self._log_model_usage_callback # <--- NEW: Register callback
        )

    @task
    def write(self) -> Task:
        return Task(
            config=self.tasks_config['write'],
            output_file='post.md',
            callback=self._log_model_usage_callback # <--- NEW: Register callback
        )
    
    @task
    def edit(self) -> Task:
        return Task(
            config=self.tasks_config['edit'],
            output_file='post.md',
            callback=self._log_model_usage_callback # <--- NEW: Register callback
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Blog crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=True,
        )