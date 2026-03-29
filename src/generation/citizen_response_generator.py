"""
Citizen-Oriented Response Generation (Phase 2.9)
Produces the final, simplified, and actionable legal awareness output for citizens.
Integrates IPC-CAM analysis, Misuse reports, and Rationale into a clear response.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# Import types from previous phases
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipc_cam.ipc_cam import IPCCAMReport
from misuse_detection.misuse_engine import MisuseReport
from rationale.legal_rationale_generator import FIRRationale


@dataclass
class CitizenResponse:
    fir_number: str
    summary_html: str
    summary_markdown: str
    language: str
    disclaimer: str = "This analysis is for informational purposes only and does not constitute legal advice. Always consult a qualified advocate."

class CitizenResponseGenerator:
    def __init__(self, 
                 api_key: str = "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",
                 model_name: str = "llama-3.3-70b-versatile",
                 use_local: bool = False,
                 local_model_path: str = "meta-llama/Llama-3.2-3B"):
        self.model_name = model_name
        self.use_local = use_local
        
        if Groq and api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None
            print("Warning: Groq client not initialized for citizen response generation.")

        # Initialize Local Model
        self.local_pipeline = None
        if use_local and pipeline:
            print(f"Loading local generative model '{local_model_path}'...")
            try:
                self.local_pipeline = pipeline(
                    "text-generation",
                    model=local_model_path,
                    device_map="auto",
                    torch_dtype="auto"
                )
            except Exception as e:
                print(f"Error loading local model: {e}")
                self.use_local = False

    def generate_full_analysis_response(self,
                                         fir_number: str,
                                         applied_sections: List[str],
                                         cam_report: IPCCAMReport,
                                         misuse_report: MisuseReport,
                                         fir_rationale: FIRRationale,
                                         language: str = "English") -> CitizenResponse:
        """
        Uses an LLM (Groq/Llama 3.3) to synthesize all technical data into a 
        citizen-friendly legal awareness report.
        """
        
        system_prompt = f"You are a legal awareness assistant helping Indian citizens understand their FIR (First Information Report) documents. You NEVER give legal advice or act as a lawyer. Your goal is to explain legal concepts in simple, clear terms and help citizens understand their rights. IMPORTANT: India has transitioned from the IPC to the Bharatiya Nyaya Sanhita (BNS). Whenever you mention an IPC section, you MUST automatically append its BNS 2024 equivalent in parentheses. Example: 'Section 420 (Now BNS Section 318)'. Always recommend consulting a qualified advocate for legal decisions. Respond in {language}. Use simple language. Avoid complex legal jargon where possible. Structure your response with clear Markdown headings."
        
        # Prepare technical context for the LLM
        tech_context = {
            "fir_number": fir_number,
            "applied_sections": applied_sections,
            "overall_misuse_risk": misuse_report.risk_assessment['risk_level'],
            "misuse_score": misuse_report.risk_assessment['overall_score'],
            "detected_misuse_patterns": misuse_report.risk_assessment['misuse_patterns'],
            "section_verdicts": fir_rationale.section_verdicts,
            "concerning_sections": fir_rationale.concerning_sections,
            "priority_actions": fir_rationale.priority_actions,
            "citizen_alert_headline": misuse_report.citizen_alert['headline']
        }
        
        class EnumEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Enum):
                    return obj.value
                return super().default(obj)

        tech_data_str = json.dumps(tech_context, indent=2, cls=EnumEncoder)

        user_prompt = f"""
Please generate a comprehensive FIR Analysis Report for a citizen based on the following technical data:

TECHNICAL DATA:
{tech_data_str}

YOUR REPORT SHOULD COVER:
1. **Executive Summary**: A high-level overview of the FIR's legal standing (is it supported by facts?).
2. **Section-by-Section Breakdown**: Explain what each applied section (e.g., Section 302, 323) means in simple terms.
3. **Key Findings**: Highlight any 'concerning' sections where the police may have overcharged the citizen (e.g., if a non-bailable section was applied without factual support).
4. **Your Rights**: List the most important rights the citizen has under the CrPC (Code of Criminal Procedure).
5. **Next Steps**: Practical, immediate actions they should take (e.g., hiring an advocate, bail applications).
6. **Questions for your Lawyer**: A few specific questions they should ask their advocate based on this analysis.

CRITICAL: Maintain a helpful but cautious tone. Do not guarantee any legal outcome.
"""

        if self.use_local and self.local_pipeline:
            try:
                # Combined prompt for local model
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
                outputs = self.local_pipeline(
                    full_prompt, 
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.3,
                    return_full_text=False
                )
                markdown_content = outputs[0]["generated_text"]
                return CitizenResponse(fir_number=fir_number, summary_html="", summary_markdown=markdown_content, language=language)
            except Exception as e:
                print(f"Local Generation Error: {e}")

        if self.client:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model_name,
                    temperature=0.3
                )

                markdown_content = chat_completion.choices[0].message.content

                return CitizenResponse(
                    fir_number=fir_number,
                    summary_html="",
                    summary_markdown=markdown_content,
                    language=language
                )

            except Exception as e:
                print(f"Error generating citizen response: {e}")

        return CitizenResponse(
            fir_number=fir_number,
            summary_html="<p>Error: LLM client not available.</p>",
            summary_markdown="Error: LLM client not available.",
            language=language
        )
# Standalone testing
if __name__ == "__main__":
    from ipc_cam.ipc_cam import IPCContextualAlignmentModule
    from rationale.legal_rationale_generator import LegalRationaleGenerator
    from misuse_detection.misuse_engine import MisuseRiskAssessmentEngine
    
    print("--- Testing Phase 2.9: Citizen-Oriented Response Generation ---")
    
    # 1. Run dependencies
    cam = IPCContextualAlignmentModule()
    rationale_gen = LegalRationaleGenerator()
    misuse_engine = MisuseRiskAssessmentEngine()
    
    # Scenario: Minor dispute overcharged as Attempt to Murder
    test_narrative = "The accused had an argument with the victim over parking. The accused slapped the victim once. The victim has no major injuries. The police applied Section 307 (Attempt to Murder)."
    test_sections = ["307"]
    fir_num = "FIR-TEST-2026-99"
    
    print("Step 1: Running Legal Analysis (CAM)...")
    cam_report = cam.generate_full_cam_report(fir_num, test_sections, test_narrative)
    fir_rationale = rationale_gen.generate_fir_level_rationale(cam_report)
    misuse_report = misuse_engine.generate_misuse_report(fir_num, cam_report, fir_rationale)
    
    # 2. Generate Final Citizen Response
    print("\nStep 2: Generating Final Citizen-Friendly Report via LLM...")
    response_gen = CitizenResponseGenerator()
    final_response = response_gen.generate_full_analysis_response(
        fir_number=fir_num,
        applied_sections=test_sections,
        cam_report=cam_report,
        misuse_report=misuse_report,
        fir_rationale=fir_rationale
    )
    
    # 3. Save
    output_path = Path("data/processed/final_reports")
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"{fir_num}_analysis.md", "w") as f:
        f.write(final_response.summary_markdown)
        
    print(f"\nSUCCESS: Report saved to {output_path}/{fir_num}_analysis.md")
    print("\n--- REPORT PREVIEW ---")
    print(final_response.summary_markdown[:500] + "...")
