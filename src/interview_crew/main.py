#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from interview_crew.crew import InterviewCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'interviewee_name': 'Mario Roman',
        'interviewer_name': 'Pradeep Ramaswamy',
        'interviewer_role': 'Director of Engineering',
        'company': 'https://www.gofundme.com/',
        'role': 'https://job-boards.greenhouse.io/gofundme/jobs/6774112',
        'interviewee_linkedin_profile': 'https://www.linkedin.com/in/romanmario/',
        'interviewer_linkedin_profile': 'https://www.linkedin.com/in/pradeepramaswamy/',
    }
    
    try:
        InterviewCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        InterviewCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        InterviewCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        InterviewCrew().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
