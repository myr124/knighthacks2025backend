import json
import hashlib
import random
from typing import Any, Dict, List

from pydantic import Field, BaseModel
from google.adk.agents import Agent, LlmAgent, ParallelAgent, SequentialAgent
from pydantic import BaseModel
from typing import List, Dict, Literal
import os
from .utils import load_instructions, format_phases_for_prompt, load_text


# Steps
# 1. Generate 50 personas of varying archetypes
# 2. Each of these personas should have some opinion on the state of the hurricane
#
class PhaseResponse(BaseModel):
    decision: Literal[
        "stay_home",
        "evacuate",
        "shelter_in_place",
        "help_neighbors",
        "gather_info",
        "wait_and_see",
    ]
    sentiment: Literal[
        "calm", "concerned", "anxious", "panicked", "skeptical", "defiant"
    ]
    location: Literal["home", "evacuating", "shelter", "with_family", "helping_others"]
    actions_taken: List[str] = Field(
        description="Bullet points of actions taken during this phase"
    )
    personality_reasoning: str = Field(
        description="1-2 sentences explaining why this persona made these decisions"
    )


class PersonaDetailedSchema(BaseModel):
    race: str
    age: int
    sex: str
    bio: str = Field(
        description="Detailed biography including occupation, family, housing, resources"
    )
    representation: float = Field(
        description="Estimated % of population this persona represents (0-100)"
    )
    response: List[PhaseResponse] = Field(
        description="Array of phase responses in chronological order (index 0 = phase 1, index 1 = phase 2, etc.)"
    )


sub_agents: List[LlmAgent] = []


# Total number of agents to generate
TOTAL_AGENTS = 50

# Archetype proportions (percentages as decimals: 0.30 = 30%, 0.05 = 5%)
# Note: These should sum to approximately 1.0 (100%)
ARCHETYPE_PROPORTIONS = {
    "lowincome_renter": 0.20,
    "middleclass_homeowner": 0.16,
    "retired_fixed_income": 0.12,
    "underemployed_gigworker": 0.10,
    "immigrant_limited_english": 0.08,
    "student_transient": 0.08,
    "highincome_professional": 0.07,
    "mobility_impaired_resident": 0.06,
    "unhoused_individual": 0.03,
    "the_planner": 0.03,
    "family_first": 0.025,
    "the_skeptic": 0.015,
    "the_anxious": 0.015,
    "information_seeker": 0.01,
}


# Archetype descriptions
ARCHETYPE_DESCRIPTIONS = {
    "lowincome_renter": "low-income, high risk, socially connected, transit-dependent, ages 25-50, shared renter household",
    "middleclass_homeowner": "middle-class, medium risk, socially active, family-focused, ages 30-55, family household",
    "retired_fixed_income": "retired, high risk, socially limited, health-conscious, ages 68-90, elderly living alone",
    "underemployed_gigworker": "underemployed, medium risk, socially average, financially_unstable, ages 22-45, shared renter household",
    "immigrant_limited_english": "low-income, high risk, socially connected, limited_english_proficiency, ages 25-60, multi-generational household",
    "student_transient": "student, low risk, socially active, transient, ages 18-24, dormitory housing",
    "highincome_professional": "high-income, low risk, socially active, proactive, ages 28-60, homeowner household",
    "mobility_impaired_resident": "mixed-income, very high risk, socially limited, medically_dependent, ages 60-90, elderly living alone",
    "unhoused_individual": "low-income, very high risk, socially isolated, shelter_in_place_preference, ages 25-65, unsheltered",
    "the_planner": "mixed-income, low risk, socially connected, proactive_planner, ages 30-60, homeowner household",
    "family_first": "mixed-income, high risk, socially connected, family_oriented, ages 28-50, family household",
    "the_skeptic": "mixed-income, medium risk, socially average, skeptical_of_authorities, ages 25-60, single adult household",
    "the_anxious": "mixed-income, medium risk, socially limited, anxious, ages 20-45, single adult household",
    "information_seeker": "mixed-income, low risk, socially active, information_seeking, ages 25-55, single adult household",
}


# Load the archetype template once and render per-archetype by replacing {ARCHETYPE_DESC} and {EMERGENCY_PHASES}
_archetype_template = load_instructions("prompts/archetype_template.txt")


# Calculate actual counts from proportions
archetype_counts = {}
agents_allocated = 0

# First pass: calculate counts using floor
for archetype, proportion in ARCHETYPE_PROPORTIONS.items():
    count = int(TOTAL_AGENTS * proportion)
    archetype_counts[archetype] = count
    agents_allocated += count

# Second pass: distribute remaining agents to archetypes with largest remainders
remaining = TOTAL_AGENTS - agents_allocated
if remaining > 0:
    remainders = [
        (archetype, (TOTAL_AGENTS * proportion) - archetype_counts[archetype])
        for archetype, proportion in ARCHETYPE_PROPORTIONS.items()
    ]
    remainders.sort(key=lambda x: x[1], reverse=True)

    for i in range(remaining):
        archetype = remainders[i][0]
        archetype_counts[archetype] += 1

# Generate agents based on calculated archetype counts
agent_counter = 0
output_keys = []
for archetype, count in archetype_counts.items():
    arche_desc = ARCHETYPE_DESCRIPTIONS[archetype]
    instruction_text = _archetype_template.replace("{ARCHETYPE_DESC}", arche_desc)

    for i in range(count):
        agent_counter += 1
        agent = LlmAgent(
            name=f"{archetype}_{i + 1}",
            model="gemini-2.0-flash",
            instruction=instruction_text,
            description=f"{archetype} population subset description",
            output_key=f"{archetype}_{i + 1}_key",
            # 👇 Enforce detailed persona output with phase responses
            output_schema=PersonaDetailedSchema,
        )
        sub_agents.append(agent)
        output_keys.append(f"{archetype}_{i + 1}_key")

# Total agents: sum of all archetype counts
print(f"Generated {agent_counter} persona agents from {TOTAL_AGENTS} requested")
print(f"Distribution: {archetype_counts}")


# Single ParallelAgent for all personas (FASTER than waves)
all_personas_agent = ParallelAgent(
    name="all_personas",
    sub_agents=sub_agents,
    description=f"All {len(sub_agents)} persona agents running in parallel",
)


root_agent = all_personas_agent
