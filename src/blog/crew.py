# Import our CrewAI LLM interceptor BEFORE importing crewai
from .crewai_llm_wrapper import InterceptedCrewAILLM

# Monkey patch BEFORE any other CrewAI imports
import crewai.llm
print("\n>>> Monkey-patching CrewAI's LLM class with our interceptor... <<<")
crewai.llm.LLM = InterceptedCrewAILLM
print(">>> Monkey-patch complete! <<<\n")

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# Additional patches for modules that might have already imported LLM
import crewai.crew
import crewai.agent
import crewai.utilities.llm_utils

# Patch the crew module's LLM reference
if hasattr(crewai.crew, 'LLM'):
    print(">>> Patching crewai.crew.LLM reference... <<<")
    crewai.crew.LLM = InterceptedCrewAILLM

# Patch any agent module references
if hasattr(crewai.agent, 'LLM'):
    print(">>> Patching crewai.agent.LLM reference... <<<")
    crewai.agent.LLM = InterceptedCrewAILLM

# Patch utilities
if hasattr(crewai.utilities.llm_utils, 'LLM'):
    print(">>> Patching crewai.utilities.llm_utils.LLM reference... <<<")
    crewai.utilities.llm_utils.LLM = InterceptedCrewAILLM

print(">>> All LLM references patched! <<<\n")

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Blog():
    """Blog crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def planner(self) -> Agent:
        return Agent(
            config=self.agents_config['planner'], # type: ignore[index]
            verbose=True,
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer'], # type: ignore[index]
            verbose=True,
        )
    
    @agent
    def editor(self) -> Agent:
        return Agent(
            config=self.agents_config['editor'], # type: ignore[index]
            verbose=True,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def plan(self) -> Task:
        return Task(
            config=self.tasks_config['plan'], # type: ignore[index]
        )

    @task
    def write(self) -> Task:
        return Task(
            config=self.tasks_config['write'], # type: ignore[index]
            output_file='post.md'
        )
    
    @task
    def edit(self) -> Task:
        return Task(
            config=self.tasks_config['edit'], # type: ignore[index]
            output_file='post.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Blog crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
