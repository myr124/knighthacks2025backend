from pydantic import Field
from google.adk.agents import Agent, LlmAgent, ParallelAgent, SequentialAgent
from pydantic import BaseModel
from typing import List
from utils import load_instructions, load_archetypes


# Steps
# 1. Generate 50 personas of varying archetypes
# 2. Each of these personas should have some opinion on the state of the hurricane
#
class PersonaMiniSchema(BaseModel):
    race: str
    age: int
    sex: str
    response: str


sub_agents: List[LlmAgent] = []

# Load personality archetypes from external JSON file
personality_archetypes = load_archetypes("multi-persona-agent/personality_archetypes.txt")

# Load the archetype prompt template once and render per-archetype by replacing {ARCHETYPE_DESC}
_archetype_prompt_template = load_instructions("multi-persona-agent/prompts/archetype_instructions_template.txt")

for archetype in personality_archetypes:
    arche_desc = personality_archetypes[archetype]
    instruction_text = _archetype_prompt_template.replace("{ARCHETYPE_DESC}", arche_desc)

    agent = LlmAgent(
        name=archetype,
        model="gemini-2.0-flash-lite",
        instruction=instruction_text,
        description=f"{archetype} population subset description",
        output_key=f"{archetype}_key",
        # 👇 Enforce tiny persona output (prevents giant blobs)
        output_schema=PersonaMiniSchema,
    )
    sub_agents.append(agent)

    # ---------------------------------------------


# 2) Single ParallelAgent for all persona agents
# ---------------------------------------------
all_personas_parallel = ParallelAgent(
    name="all_personas",
    sub_agents=sub_agents,
    description=f"All {len(sub_agents)} persona agents running in parallel",
)


# --- Output schema for the merger (the big final object you already use) ---
class outputSchema(BaseModel):
    output: str = Field(description="test")


# --- Merger Agent ---
# TIP in your synthesis_prompt.txt:
# - Read only persona keys that end with "_review".
# - Ignore any missing keys (some personas may fail).
# - Produce exactly ONE final JSON into output_key="final_summary".
merger_agent = LlmAgent(
    name="merger_agent",
    model="gemini-2.5-flash-lite",
    instruction=load_instructions("multi-persona-agent/prompts/summarizer_instructions.txt"),
    description="Summarizes the entire sentiment of all population subsets",
    output_schema=outputSchema,
    output_key="final_summary",
)

# --- Sequential Pipeline ---
# Phase 1: All personas run in parallel
# Phase 2: Merger aggregates results
sequential_pipeline_agent = SequentialAgent(
    name="FinalAnalysisAgent",
    sub_agents=[
        all_personas_parallel,  # Phase 1: all personas in parallel
        merger_agent,           # Phase 2: aggregation
    ],
    description="Coordinates the creation of agents and summarization of reactions to hurricane",
)


root_agent = sequential_pipeline_agent
