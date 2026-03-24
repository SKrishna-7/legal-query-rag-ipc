"""
Legal Rationale Generator (Phase 2.7) — CORE NOVELTY
Translates the numerical and algorithmic outputs of the IPC-CAM module
into human-readable, structured explainable AI reports for both
legal professionals and citizens.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

# Import the types from Phase 2.4 (IPC-CAM)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipc_cam.ipc_cam import (
    SectionAlignmentResult, 
    IPCCAMReport,
    SatisfactionStatus,
    AlignmentStatus,
    IngredientScore
)


@dataclass
class SectionRationale:
    section: str
    title: str
    alignment_verdict: str
    alignment_score: float
    ingredient_analysis: List[Dict[str, Any]]
    missing_elements: List[str]
    verdict_reasoning: str
    recommended_sections: List[str]
    citizen_explanation: str
    legal_disclaimer: str = "This analysis is for informational purposes only and does not constitute legal advice. Consult a qualified advocate for professional legal assistance."

@dataclass
class FIRRationale:
    fir_number: str
    overall_misuse_risk: str
    risk_score: float
    section_verdicts: List[Dict[str, str]]
    concerning_sections: List[Dict[str, str]]
    correctly_applied_sections: List[str]
    priority_actions: List[str]

class LegalRationaleGenerator:
    def __init__(self, ipc_kb_path: str = "data/processed/ipc_sections/ipc_complete.json"):
        self.ipc_kb_path = ipc_kb_path
        self.ipc_kb = self._load_ipc_kb()

    def _load_ipc_kb(self) -> Dict[str, Any]:
        if not Path(self.ipc_kb_path).exists():
            print(f"Warning: IPC KB not found at {self.ipc_kb_path}")
            return {}
        with open(self.ipc_kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return {str(item.get("section_number")): item for item in data if item.get("section_number")}
            elif isinstance(data, dict):
                return data
        return {}

    def generate_section_rationale(self, 
                                   section_number: str, 
                                   alignment_result: SectionAlignmentResult, 
                                   fir_narrative: str) -> SectionRationale:
        """
        Generates a structured, explainable rationale for a single IPC section
        based on the IPC-CAM alignment output.
        """
        kb_data = self.ipc_kb.get(section_number, {})
        title = kb_data.get("title", f"Section {section_number}")
        
        ingredient_analysis = []
        for ing in alignment_result.ingredient_scores:
            ingredient_analysis.append({
                "ingredient": ing.ingredient,
                "status": ing.satisfaction_status.value if isinstance(ing.satisfaction_status, Enum) else str(ing.satisfaction_status),
                "evidence": ing.evidence_sentences,
                "confidence": round(ing.confidence, 2),
                "legal_note": ing.reasoning if ing.reasoning else "Analyzed via NLP embedding/entailment"
            })

        # Generate verdict reasoning based on status
        if alignment_result.alignment_status == AlignmentStatus.FULLY_ALIGNED:
            reasoning = f"Section {section_number} ({title}) appears correctly applied. All essential legal ingredients are supported by the facts stated in the FIR narrative."
            citizen_exp = f"The police charged you under Section {section_number} ({title}). Based on the facts in the FIR, this charge seems to match the legal requirements."
        elif alignment_result.alignment_status == AlignmentStatus.PARTIALLY_ALIGNED:
            reasoning = f"Section {section_number} is only partially supported. While some facts align, it lacks clear evidence for: {', '.join(alignment_result.missing_ingredients[:2])}."
            citizen_exp = f"You were charged under Section {section_number}, but the FIR facts might not fully support this severe charge. Your lawyer could argue that essential elements are missing."
        else:
            reasoning = f"Potential Misapplication: The facts in the FIR do not support the essential ingredients for Section {section_number}. Specifically lacking: {', '.join(alignment_result.missing_ingredients)}."
            citizen_exp = f"Warning: The charge under Section {section_number} ({title}) appears to be unsupported by the facts written in the FIR. This could be an overcharge. Your lawyer should strongly contest this."

        # Simple recommendation heuristic (In a full system, this would call Phase 2.6/LLM)
        recommended = []
        if "murder" in title.lower() and alignment_result.alignment_status != AlignmentStatus.FULLY_ALIGNED:
            recommended = ["304", "304A"] # Suggest Culpable Homicide instead
        if "theft" in title.lower() and alignment_result.alignment_status != AlignmentStatus.FULLY_ALIGNED:
            recommended = ["403", "411"]

        return SectionRationale(
            section=section_number,
            title=title,
            alignment_verdict=alignment_result.alignment_status.value if isinstance(alignment_result.alignment_status, Enum) else str(alignment_result.alignment_status),
            alignment_score=round(alignment_result.alignment_score, 2),
            ingredient_analysis=ingredient_analysis,
            missing_elements=alignment_result.missing_ingredients,
            verdict_reasoning=reasoning,
            recommended_sections=recommended,
            citizen_explanation=citizen_exp
        )

    def generate_fir_level_rationale(self, cam_report: IPCCAMReport) -> FIRRationale:
        """
        Aggregates multiple section rationales into a comprehensive FIR-level summary.
        """
        section_verdicts = []
        concerning = []
        correctly_applied = []
        
        for res in cam_report.sections_evaluated:
            status_val = res.alignment_status.value if isinstance(res.alignment_status, Enum) else str(res.alignment_status)
            section_verdicts.append({
                "section": res.section_number,
                "status": status_val,
                "score": f"{res.alignment_score:.2f}"
            })
            
            if res.alignment_status == AlignmentStatus.MISALIGNED or res.alignment_status == AlignmentStatus.PARTIALLY_ALIGNED:
                concerning.append({
                    "section": res.section_number,
                    "reason": res.alignment_reasoning
                })
            else:
                correctly_applied.append(res.section_number)

        # Risk Classification
        score = cam_report.overall_misuse_risk_score
        if score > 0.66:
            risk_level = "HIGH RISK OF MISUSE / OVERCHARGING"
        elif score > 0.33:
            risk_level = "MEDIUM RISK (Partial Support)"
        else:
            risk_level = "LOW RISK (Charges Legally Supported)"

        # Priority Actions
        actions = ["Consult a qualified advocate immediately."]
        if score > 0.5:
            actions.append("Consider filing for Anticipatory Bail, as severe charges lack factual backing.")
            actions.append("Prepare to file a petition under Section 482 CrPC to quash the unsupported charges.")
        else:
            actions.append("Prepare defense based on the facts clearly established in the FIR.")

        return FIRRationale(
            fir_number=cam_report.fir_number,
            overall_misuse_risk=risk_level,
            risk_score=round(score, 2),
            section_verdicts=section_verdicts,
            concerning_sections=concerning,
            correctly_applied_sections=correctly_applied,
            priority_actions=actions
        )

    def generate_element_satisfaction_visualization(self, section_rationale: SectionRationale) -> Dict[str, Any]:
        """
        Generates data for a radar/spider chart visualization on the frontend.
        """
        labels = []
        data = []
        
        for ing in section_rationale.ingredient_analysis:
            labels.append(ing["ingredient"][:30] + "...") # Truncate long labels
            if ing["status"] == "SATISFIED":
                data.append(1.0)
            elif ing["status"] == "PARTIALLY_SATISFIED":
                data.append(0.5)
            else:
                data.append(0.0)
                
        return {
            "chart_type": "radar",
            "section": section_rationale.section,
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Ingredient Satisfaction Score",
                    "data": data,
                    "fill": True,
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "borderColor": "rgb(54, 162, 235)",
                    "pointBackgroundColor": "rgb(54, 162, 235)",
                }]
            }
        }

# Standalone testing
if __name__ == "__main__":
    from ipc_cam.ipc_cam import IPCContextualAlignmentModule
    
    print("--- Testing Phase 2.7: Legal Rationale Generator ---")
    
    # 1. Run a mock CAM analysis first
    cam = IPCContextualAlignmentModule()
    # A narrative missing the "intention" required for murder
    test_narrative = "The accused was driving recklessly and hit the victim, causing immediate death. The accused then fled the scene."
    test_sections = ["302"] # Applied Murder for a rash driving case (Classic overcharge)
    
    print("Running IPC-CAM Analysis...")
    cam_report = cam.generate_full_cam_report("FIR-2026-001", test_sections, test_narrative)
    
    # 2. Generate Rationales
    print("\nGenerating Legal Rationales...")
    rationale_gen = LegalRationaleGenerator()
    
    # Section Rationale
    sec_rationale = rationale_gen.generate_section_rationale(
        section_number=test_sections[0],
        alignment_result=cam_report.sections_evaluated[0],
        fir_narrative=test_narrative
    )
    
    # FIR Rationale
    fir_rationale = rationale_gen.generate_fir_level_rationale(cam_report)
    
    # Visual Data
    vis_data = rationale_gen.generate_element_satisfaction_visualization(sec_rationale)
    
    # Save Outputs
    output_dir = Path("data/processed/rationale")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "section_rationale.json", "w") as f:
        json.dump(asdict(sec_rationale), f, indent=2)
        
    with open(output_dir / "fir_rationale.json", "w") as f:
        json.dump(asdict(fir_rationale), f, indent=2)
        
    with open(output_dir / "visual_data.json", "w") as f:
        json.dump(vis_data, f, indent=2)
        
    print("\n--- Summary ---")
    print(f"FIR Risk Level: {fir_rationale.overall_misuse_risk}")
    print(f"Section 302 Verdict: {sec_rationale.alignment_verdict}")
    print("\nCitizen Explanation:")
    print(sec_rationale.citizen_explanation)
    print("\nOutputs saved to data/processed/rationale/")
