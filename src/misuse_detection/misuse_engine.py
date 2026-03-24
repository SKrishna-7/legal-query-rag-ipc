"""
Misuse Risk Assessment Engine (Phase 2.8) — CORE NOVELTY
Computes the final misuse risk score based on IPC-CAM outputs,
detects specific patterns of police overcharging/undercharging,
and generates actionable alerts for citizens.
"""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path

# Import types from Phase 2.4 and 2.7
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipc_cam.ipc_cam import (
    SectionAlignmentResult,
    IPCCAMReport,
    AlignmentStatus,
    SatisfactionStatus
)
from rationale.legal_rationale_generator import FIRRationale


class MisuseType(Enum):
    OVER_SEVERITY = 'More serious section than facts support'
    UNDER_SEVERITY = 'Less serious section, likely to compromise prosecution'
    MISSING_MENS_REA = 'Section requires criminal intent not shown in facts'
    MISSING_ACTUS_REUS = 'Section requires specific act not described in FIR'
    WRONG_JURISDICTION = 'Section not applicable in this jurisdiction'
    PROCEDURAL_MISMATCH = 'Cognizable/non-cognizable classification error'
    COMPANION_SECTION_MISUSE = 'Joint liability section (34/120B) wrongly applied'
    BAIL_MANIPULATION = 'Non-bailable section applied to deny bail inappropriately'

class RiskLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class DetectedPattern:
    misuse_type: MisuseType
    severity_multiplier: float
    affected_sections: List[str]
    description: str

@dataclass
class MisuseRiskScore:
    overall_score: float
    risk_level: RiskLevel
    misuse_patterns: List[Dict[str, Any]]  # Serialized DetectedPattern
    section_risk_scores: Dict[str, float]
    confidence_interval: Tuple[float, float]
    risk_reasoning: str

@dataclass
class CitizenAlert:
    alert_level: str
    headline: str
    challenge_sections: List[Dict[str, str]]
    immediate_actions: List[str]
    crpc_rights: List[str]
    questions_for_advocate: List[str]

@dataclass
class MisuseReport:
    fir_number: str
    risk_assessment: Dict[str, Any]  # Serialized MisuseRiskScore
    citizen_alert: Dict[str, Any]    # Serialized CitizenAlert
    cam_summary: str
    rationale_summary: Dict[str, Any]


class MisuseRiskAssessmentEngine:
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

    def _detect_patterns(self, cam_report: IPCCAMReport) -> List[DetectedPattern]:
        patterns = []
        
        for section_res in cam_report.sections_evaluated:
            sec_num = section_res.section_number
            kb_data = self.ipc_kb.get(sec_num, {})
            
            # 1. Companion Section Misuse (e.g., 34, 120B)
            if sec_num in ["34", "120B"] and section_res.alignment_status != AlignmentStatus.FULLY_ALIGNED:
                patterns.append(DetectedPattern(
                    misuse_type=MisuseType.COMPANION_SECTION_MISUSE,
                    severity_multiplier=1.2,
                    affected_sections=[sec_num],
                    description=f"Section {sec_num} (Joint Liability/Conspiracy) applied without clear evidence of meeting of minds or common intention."
                ))
                
            # 2. Missing Mens Rea vs Actus Reus
            missing_mens_rea = False
            missing_actus_reus = False
            
            for ing in section_res.ingredient_scores:
                if ing.satisfaction_status != SatisfactionStatus.SATISFIED:
                    ing_lower = ing.ingredient.lower()
                    if any(word in ing_lower for word in ["intention", "knowledge", "dishonestly", "fraudulently", "knowing"]):
                        missing_mens_rea = True
                    else:
                        missing_actus_reus = True
                        
            if missing_mens_rea:
                patterns.append(DetectedPattern(
                    misuse_type=MisuseType.MISSING_MENS_REA,
                    severity_multiplier=1.5,
                    affected_sections=[sec_num],
                    description=f"Section {sec_num} requires specific criminal intent (Mens Rea) which is not established in the FIR facts."
                ))
            if missing_actus_reus:
                patterns.append(DetectedPattern(
                    misuse_type=MisuseType.MISSING_ACTUS_REUS,
                    severity_multiplier=1.3,
                    affected_sections=[sec_num],
                    description=f"The physical act (Actus Reus) required for Section {sec_num} is missing or partially described in the FIR."
                ))
                
            # 3. Bail Manipulation (Non-bailable section with poor alignment)
            if not kb_data.get("bailable", True) and section_res.alignment_status == AlignmentStatus.MISALIGNED:
                patterns.append(DetectedPattern(
                    misuse_type=MisuseType.BAIL_MANIPULATION,
                    severity_multiplier=2.0,
                    affected_sections=[sec_num],
                    description=f"Severe non-bailable Section {sec_num} applied despite poor factual alignment, potentially to deny bail."
                ))
                
            # 4. General Over-severity
            if section_res.alignment_status == AlignmentStatus.MISALIGNED:
                patterns.append(DetectedPattern(
                    misuse_type=MisuseType.OVER_SEVERITY,
                    severity_multiplier=1.0,
                    affected_sections=[sec_num],
                    description=f"Section {sec_num} applied but facts do not support it. Likely an overcharge."
                ))
                
        return patterns

    def compute_misuse_risk_score(self, cam_report: IPCCAMReport, fir_narrative: str = "") -> MisuseRiskScore:
        """Computes the final misuse risk score based on detected patterns."""
        
        weights = {
            MisuseType.OVER_SEVERITY: 0.30,
            MisuseType.MISSING_MENS_REA: 0.25,
            MisuseType.MISSING_ACTUS_REUS: 0.20,
            MisuseType.PROCEDURAL_MISMATCH: 0.15,
            MisuseType.COMPANION_SECTION_MISUSE: 0.10,
            MisuseType.BAIL_MANIPULATION: 0.40,
            MisuseType.UNDER_SEVERITY: 0.20,
            MisuseType.WRONG_JURISDICTION: 0.15
        }
        
        detected_patterns = self._detect_patterns(cam_report)
        
        base_score = 0.0
        section_risk_scores = {}
        
        for pattern in detected_patterns:
            score_addition = weights.get(pattern.misuse_type, 0.1) * pattern.severity_multiplier
            base_score += score_addition
            for sec in pattern.affected_sections:
                section_risk_scores[sec] = section_risk_scores.get(sec, 0.0) + score_addition
                
        # Alignment penalty
        if cam_report.sections_evaluated:
            avg_alignment = sum(s.alignment_score for s in cam_report.sections_evaluated) / len(cam_report.sections_evaluated)
        else:
            avg_alignment = 0.0
            
        alignment_penalty = 1.0 - avg_alignment
        
        # Final calculation
        final_score = min(1.0, (base_score * 0.7) + (alignment_penalty * 0.3))
        
        # Assign Level
        if final_score >= 0.70:
            risk_level = RiskLevel.HIGH
            reasoning = "High risk of FIR misuse. Multiple severe discrepancies detected between the applied legal sections and the narrative facts, including potential bail manipulation or missing criminal intent."
        elif final_score >= 0.40:
            risk_level = RiskLevel.MEDIUM
            reasoning = "Medium risk of FIR misuse. Some sections are only partially supported by facts, or companion sections were applied loosely. Warrants legal review."
        else:
            risk_level = RiskLevel.LOW
            reasoning = "Low risk of misuse. The legal sections applied generally align well with the facts stated in the FIR narrative."

        # Serialize patterns
        serialized_patterns = []
        for p in detected_patterns:
            serialized_patterns.append({
                "misuse_type": p.misuse_type.value,
                "severity_multiplier": p.severity_multiplier,
                "affected_sections": p.affected_sections,
                "description": p.description
            })

        return MisuseRiskScore(
            overall_score=round(final_score, 2),
            risk_level=risk_level,
            misuse_patterns=serialized_patterns,
            section_risk_scores={k: round(v, 2) for k, v in section_risk_scores.items()},
            confidence_interval=(max(0.0, final_score - 0.1), min(1.0, final_score + 0.1)),
            risk_reasoning=reasoning
        )

    def generate_citizen_alert(self, misuse_risk: MisuseRiskScore) -> CitizenAlert:
        """Generates actionable alerts tailored to the citizen's risk level."""
        
        challenge_sections = []
        for pattern in misuse_risk.misuse_patterns:
            for sec in pattern["affected_sections"]:
                challenge_sections.append({
                    "section": sec,
                    "reason": pattern["description"]
                })
                
        # Deduplicate
        unique_challenges = []
        seen = set()
        for c in challenge_sections:
            if c["section"] not in seen:
                unique_challenges.append(c)
                seen.add(c["section"])

        crpc_rights = [
            "Section 41D CrPC: Right to meet an advocate of your choice during interrogation.",
            "Section 50 CrPC: Person arrested to be informed of grounds of arrest and of right to bail."
        ]
        
        if misuse_risk.risk_level == RiskLevel.HIGH:
            headline = "⚠️ HIGH ALERT: Potential Legal Overcharging Detected"
            actions = [
                "URGENT: Do not make any statements to the police without a lawyer present.",
                "URGENT: File an application for Anticipatory Bail immediately (Section 438 CrPC).",
                "Consider filing a petition under Section 482 CrPC in the High Court to quash the unsupported charges."
            ]
            questions = [
                "Can we file for anticipatory bail immediately based on the lack of evidence for the non-bailable charges?",
                "Should we file a Section 482 petition to quash the severe charges that lack factual backing in the FIR?",
                "How do we legally document that the 'mens rea' (intention) is missing from their accusations?"
            ]
        elif misuse_risk.risk_level == RiskLevel.MEDIUM:
            headline = "⚠️ NOTICE: Partial Discrepancies in FIR Charges"
            actions = [
                "Consult a lawyer to review the specific sections flagged.",
                "Gather evidence (CCTV, witnesses, documents) that contradicts the unsupported claims.",
                "Prepare a standard bail application."
            ]
            questions = [
                "The system flagged some sections as 'partially aligned'. How can we use this to weaken their case?",
                "Are the charges applied cognizable or non-cognizable?",
                "Should we apply for regular bail immediately?"
            ]
        else:
            headline = "ℹ️ INFO: Charges Appear Legally Aligned with FIR Narrative"
            actions = [
                "Consult a lawyer to prepare your general defense.",
                "Cooperate with the legal process while maintaining your right to silence regarding self-incrimination."
            ]
            questions = [
                "What is the standard punishment if convicted under these sections?",
                "What is our best defense strategy given that the FIR narrative structurally supports the charges?",
                "How long does the trial process typically take for these sections?"
            ]

        return CitizenAlert(
            alert_level=misuse_risk.risk_level.value,
            headline=headline,
            challenge_sections=unique_challenges,
            immediate_actions=actions,
            crpc_rights=crpc_rights,
            questions_for_advocate=questions
        )

    def generate_misuse_report(self, 
                               fir_number: str, 
                               cam_report: IPCCAMReport, 
                               fir_rationale: FIRRationale) -> MisuseReport:
        """Generates the final comprehensive Misuse Assessment Report."""
        
        misuse_risk = self.compute_misuse_risk_score(cam_report)
        citizen_alert = self.generate_citizen_alert(misuse_risk)
        
        return MisuseReport(
            fir_number=fir_number,
            risk_assessment=asdict(misuse_risk),
            citizen_alert=asdict(citizen_alert),
            cam_summary=cam_report.cam_summary,
            rationale_summary={
                "overall_risk": fir_rationale.overall_misuse_risk,
                "concerning_sections": fir_rationale.concerning_sections
            }
        )

# Standalone testing
if __name__ == "__main__":
    from ipc_cam.ipc_cam import IPCContextualAlignmentModule
    from rationale.legal_rationale_generator import LegalRationaleGenerator
    
    print("--- Testing Phase 2.8: Misuse Risk Assessment Engine ---")
    
    # 1. Run dependencies
    cam = IPCContextualAlignmentModule()
    rationale_gen = LegalRationaleGenerator()
    
    # Test Scenario: Slapping someone (323) but police applied Attempt to Murder (307)
    test_narrative = "The accused had an argument with the victim over parking. In a fit of anger, the accused slapped the victim across the face once, causing a minor bruise. The accused then walked away."
    test_sections = ["307"] # Attempt to murder (Classic massive overcharge / bail manipulation)
    
    print("Running IPC-CAM Analysis...")
    cam_report = cam.generate_full_cam_report("FIR-2026-002", test_sections, test_narrative)
    fir_rationale = rationale_gen.generate_fir_level_rationale(cam_report)
    
    # 2. Run Misuse Engine
    print("\nRunning Misuse Risk Engine...")
    misuse_engine = MisuseRiskAssessmentEngine()
    
    report = misuse_engine.generate_misuse_report(
        fir_number="FIR-2026-002",
        cam_report=cam_report,
        fir_rationale=fir_rationale
    )
    
    # 3. Save Output
    output_dir = Path("data/processed/misuse_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class EnumEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Enum):
                return obj.value
            return super().default(obj)
            
    with open(output_dir / "misuse_report.json", "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, cls=EnumEncoder)
        
    print("\n--- Summary ---")
    print(f"Final Risk Score: {report.risk_assessment['overall_score']}")
    print(f"Risk Level: {report.risk_assessment['risk_level']}")
    print("\nDetected Patterns:")
    for p in report.risk_assessment['misuse_patterns']:
        print(f"- [{p['misuse_type']}] {p['description']}")
    print("\nCitizen Alert Headline:")
    print(report.citizen_alert['headline'])
    print("\nImmediate Actions:")
    for a in report.citizen_alert['immediate_actions']:
        print(f"- {a}")
    print("\nOutputs saved to data/processed/misuse_reports/")
