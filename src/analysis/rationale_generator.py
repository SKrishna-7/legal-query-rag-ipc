"""
Legal IPC-RAG — Legal Rationale Generator (Phase 2.7)
=====================================================
Transforms structured json outputs from IPC-CAM and Misuse Detection
into professional, human-readable legal audit reports.
"""

import json
import logging
from typing import Dict
from groq import Groq

logger = logging.getLogger(__name__)

class LegalRationaleGenerator:
    """
    Generates explainable, human-readable legal reports.
    """
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate_report(self, analyzed_fir: Dict) -> str:
        """
        Takes a fully analyzed FIR dict (from IPC-CAM + Misuse Engine)
        and generates a Markdown report.
        """
        prompt = f"""
        You are an expert Indian Legal Consultant. Review the following structured Consistency Analysis Report for an FIR and draft a highly professional Legal Audit Report in Markdown format.

        DATA:
        {json.dumps(analyzed_fir, indent=2)}

        INSTRUCTIONS FOR THE REPORT:
        1. Use a formal, objective, and legally sound tone.
        2. Structure the report exactly as follows:
            # Legal Consistency Audit Report
            **FIR Number:** [fir_number] | **Overall Consistency Score:** [score]/1.0
            
            ## 1. Executive Summary
            [Summarize the overall findings, explicitly stating if there is a potential legal misuse or inconsistency.]
            
            ## 2. Detailed Section Analysis
            [For each section, create a subsection:]
            ### Section [Number]
            * **Assessment:** [Overall assessment text]
            * **Misuse Classification:** [Category] - [Description]
            * **Satisfied Ingredients:** [List them with evidence]
            * **Missing Legal Elements:** [Crucial part: List missing ingredients and why their absence is problematic legally]
            
            ## 3. Legal Recommendations
            [Based on the missing elements and misuse classification, provide 2-3 actionable recommendations for the defense lawyer or the auditing officer.]

        Generate ONLY the Markdown text. Do not include any conversational filler.
        """

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Rationale Generation failed: {e}")
            return f"Error generating report: {e}"

if __name__ == "__main__":
    print("Rationale Generator Module Loaded Successfully.")
