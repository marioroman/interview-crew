from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import WebsiteSearchTool, SpiderTool, PDFSearchTool
import os
import dotenv
from crewai.llm import LLM

dotenv.load_dotenv()

web_search_tool = WebsiteSearchTool()
web_scraper = SpiderTool()
resume_tool = PDFSearchTool(pdf="/Users/marioroman/sources/interview_crew/src/interview_crew/docs/mario-roman-resume-v2.pdf")
preparation_data_tool = PDFSearchTool(pdf="/Users/marioroman/sources/interview_crew/src/interview_crew/docs/combined-interview-prep-guide_v1.pdf")
interviewer_resume_tool = PDFSearchTool(pdf="/Users/marioroman/sources/interview_crew/src/interview_crew/docs/pradeep.pdf")

@CrewBase
class InterviewCrew():
    """InterviewCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def people_background_research(self) -> Agent:
        return Agent(
            config=self.agents_config['people_background_research'], # type: ignore[index]
            verbose=True,
            tools=[web_scraper, resume_tool, web_search_tool, interviewer_resume_tool],
        )

    @agent
    def interviewee(self) -> Agent:
        return Agent(
            config=self.agents_config['interviewee'], # type: ignore[index]
            verbose=True,   
            tools=[web_scraper, preparation_data_tool, web_search_tool],
        )

    @task
    def people_background_research_interviewee_task(self) -> Task:
        return Task(
            config=self.tasks_config['people_background_research_interviewee_task'], # type: ignore[index]
        )

    @task
    def people_background_research_interviewer_task(self) -> Task:
        return Task(
            config=self.tasks_config['people_background_research_interviewer_task'], # type: ignore[index]
        )

    @task
    def interviewee_task(self) -> Task:
        return Task(
            config=self.tasks_config['interviewee_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the InterviewCrew crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            planning=True,
            #process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
            manager_llm=LLM(model="openai/gpt-4o"),
        )
