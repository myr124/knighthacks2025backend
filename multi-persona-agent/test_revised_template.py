"""
Test script to verify revised template with archetype influence field.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_text, load_json

def test_revised_template():
    """Test the revised template with all placeholders."""
    print("=" * 80)
    print("REVISED TEMPLATE TEST - Archetype Influence Integration")
    print("=" * 80)

    # Load template
    print("\n[1/5] Loading revised template...")
    template = load_text("multi-persona-agent/prompts/archetype_instructions_template.txt")
    print(f"[OK] Loaded template ({len(template)} characters)")

    # Check for all required placeholders
    print("\n[2/5] Checking for required placeholders...")
    required_placeholders = [
        "{EMERGENCY_PLAN}",
        "{ARCHETYPE_DESC}",
        "{PHASE_NAME}",
        "{PHASE_CONTENT}"
    ]

    all_found = True
    for placeholder in required_placeholders:
        if placeholder in template:
            print(f"[OK] {placeholder} found")
        else:
            print(f"[ERROR] {placeholder} NOT found")
            all_found = False

    if not all_found:
        return False

    # Check for archetype_influence in output format
    print("\n[3/5] Checking for archetype_influence field...")
    if '"archetype_influence": str' in template:
        print("[OK] archetype_influence field found in output schema")
    else:
        print("[ERROR] archetype_influence field NOT found in output schema")
        return False

    # Load real data and test replacement
    print("\n[4/5] Testing with real data...")
    emergency_plan = load_text("multi-persona-agent/prompts/emergency_manager_12hour_plan.txt")
    archetypes = load_json("multi-persona-agent/prompts/personality_archetypes.txt")

    test_archetype = "themeparkworker"
    test_phase = "T-48 HOURS: Tuesday 6:00 AM"
    test_content = "HURRICANE WARNING ISSUED - Storm maintaining Category 4 strength..."

    # Replace all placeholders
    result = (
        template
        .replace("{EMERGENCY_PLAN}", emergency_plan)
        .replace("{ARCHETYPE_DESC}", archetypes[test_archetype])
        .replace("{PHASE_NAME}", test_phase)
        .replace("{PHASE_CONTENT}", test_content)
    )

    # Verify no placeholders remain
    for placeholder in required_placeholders:
        if placeholder in result:
            print(f"[ERROR] {placeholder} still present after replacement")
            return False

    print(f"[OK] All placeholders replaced successfully")
    print(f"[OK] Final instruction length: {len(result)} characters")

    # Verify key content is present
    print("\n[5/5] Verifying key content...")
    checks = [
        ("Emergency plan content", "T-72 HOURS"),
        ("Archetype description", archetypes[test_archetype][:30]),
        ("Phase name", test_phase),
        ("Output field: archetype_influence", '"archetype_influence"'),
        ("Output field: current_location", '"current_location"'),
        ("Output field: resources_remaining", '"resources_remaining"'),
        ("Instruction about archetype", "archetype characteristics SPECIFICALLY shaped"),
    ]

    all_checks_passed = True
    for check_name, check_text in checks:
        if check_text in result:
            print(f"[OK] {check_name} present")
        else:
            print(f"[ERROR] {check_name} NOT present")
            all_checks_passed = False

    if not all_checks_passed:
        return False

    # Show sample
    print("\n" + "=" * 80)
    print("SAMPLE OF GENERATED INSTRUCTION (first 800 chars):")
    print("=" * 80)
    print(result[:800] + "...")
    print("\n" + "=" * 80)
    print("SAMPLE SHOWING ARCHETYPE_INFLUENCE SECTION:")
    print("=" * 80)

    # Find and display archetype_influence section
    influence_start = result.find("archetype_influence")
    if influence_start > 0:
        print(result[influence_start:influence_start+600] + "...")

    print("\n" + "=" * 80)
    print("TEST PASSED - Revised template working correctly!")
    print("=" * 80)
    print("\nTemplate now includes:")
    print("  ✓ Full emergency plan context")
    print("  ✓ Archetype description integration")
    print("  ✓ Detailed state tracking fields")
    print("  ✓ NEW: archetype_influence field")
    print("  ✓ Explicit instructions to connect archetype to outcomes")

    return True


if __name__ == "__main__":
    success = test_revised_template()
    sys.exit(0 if success else 1)
