from pydantic import Field
from google.adk.agents import Agent, LlmAgent, ParallelAgent, SequentialAgent
from pydantic import BaseModel
from typing import List, Dict, Any
from utils import load_json, load_text, parse_emergency_plan_phases


# Steps
# 1. Generate personas of varying archetypes
# 2. Each persona responds to each phase of the emergency plan
# 3. Aggregate responses per phase
#
class PersonaMiniSchema(BaseModel):
    race: str
    age: int
    sex: str
    response: str


class PhaseResponseSchema(BaseModel):
    """Output schema for a single phase response from one persona."""
    phase_name: str = Field(description="Name of the emergency phase")
    race: str
    age: int
    sex: str
    response: str


class PersonaStateTracking(BaseModel):
    """Output schema for 12-hour interval tracking with state persistence."""
    race: str = Field(description="Demographic - consistent across all periods")
    age: int = Field(description="Age - consistent across all periods")
    sex: str = Field(description="Sex - consistent across all periods")
    current_location: str = Field(description="Where the person is right now")
    action_taken_this_period: str = Field(description="What they did in the last 12 hours (40-60 words)")
    current_emotional_state: str = Field(description="How they feel right now (20-30 words)")
    resources_remaining: str = Field(description="Current resource status (30-40 words)")
    physical_condition: str = Field(description="Physical state right now (20-30 words)")
    main_concern_next_12hrs: str = Field(description="Primary worry for next 12-hour period (25-35 words)")


sub_agents: List[LlmAgent] = []

# Load personality archetypes from external JSON file
personality_archetypes = {
  "lowincome": "low-income, high risk, socially connected",
  "middleclass": "middle-class, low risk, socially average",
  "retired": "retired, high risk, socially limited",
  "underemployed": "under-employed, medium risk, socially connected",
  "highincome": "high-income, medium risk, socially active",
  "student": "student, low risk, socially connected",
  "themeparkworker": "theme park employee, low-medium income, high social exposure, service-oriented",
  "hospitalityworker": "hospitality/hotel worker, low income, multilingual, tourism-dependent",
  "seasonaltourist": "seasonal tourism worker, high financial volatility, transient social ties",
  "latinofamily": "Latino family head, medium income, strong community ties, bilingual household",
  "caribbeanimmigrant": "Caribbean immigrant, medium-low income, dual-culture connected, remittance-sender",
  "ucfstudent": "UCF student, low income, highly social, tech-savvy, part-time employed",
  "floridaretiree": "Florida retiree, fixed income, healthcare-dependent, snowbird social",
  "conventionworker": "convention center worker, irregular income, event-dependent, customer-facing",
  "restaurantworker": "restaurant worker, tip-dependent income, multiple jobs, socially exhausted",
  "airportemployee": "Orlando airport employee, shift worker, diverse coworker network, travel benefits",
  "entertainmentcast": "entertainment/performer, irregular income, artistic community, gig-based",
  "vacationrental": "vacation rental owner/manager, entrepreneurial, property-dependent income",
  "spacecoastcommuter": "aerospace industry commuter, high income, STEM-educated, geographically split",
  "militaryfamily": "military family member, mobile lifestyle, government benefits, base-connected",
  "medicalworker": "healthcare worker, medium-high income, high stress, essential worker",
  "constructionworker": "construction worker, boom-bust income, physically demanding, weather-dependent",
  "collegegrad": "recent UCF grad, early career, tech industry hopeful, socially active",
  "suburbanfamily": "suburban family, dual income, child-focused, community-involved",
  "internationaltourist": "international worker/student, temporary visa, culturally isolated, goal-oriented",
  "downtownprofessional": "downtown Orlando professional, urban lifestyle, young professional network"
}


# Load the archetype prompt template once and render per-archetype by replacing {ARCHETYPE_DESC}
_archetype_prompt_template = load_text("multi-persona-agent/prompts/archetype_instructions_template.txt")

# Load emergency plan 
_emergency_plan = load_text("multi-persona-agent/prompts/emergency_plan.txt")

for archetype in personality_archetypes:
    arche_desc = personality_archetypes[archetype]
    instruction_text = _archetype_prompt_template.replace("{ARCHETYPE_DESC}", arche_desc).replace("{EMERGENCY_PLAN}", _emergency_plan)

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
    instruction=load_text("multi-persona-agent/prompts/summarizer_instructions.txt"),
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
