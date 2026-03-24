"""
Legal IPC-RAG — IPC Consistency Analysis Module (IPC-CAM)
==========================================================
Analyzes the consistency between FIR narratives and the essential 
ingredients of applied IPC sections. Core novelty of the proposed system.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IPCCAMModule:
    """
    Consistency Analysis Module to check if FIR facts satisfy IPC ingredients.
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.kb_path = "data/processed/ipc_sections/ipc_complete.json"
        self.ipc_kb = self._load_kb()

    def _load_kb(self) -> Dict[str, Dict]:
        """Load the enriched IPC Knowledge Base."""
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Map by section number for fast retrieval
                return {str(item['section_number']): item for item in data}
        except Exception as e:
            logger.error(f"Failed to load IPC KB: {e}")
            return {}

    def analyze_consistency(self, fir_narrative: str, section_number: str) -> Dict:
        """
        Analyze consistency between facts and a specific IPC section.
        """
        section_data = self.ipc_kb.get(str(section_number))
        
        if not section_data:
            return {
                "section": section_number,
                "status": "error",
                "reason": "Section details not found in Knowledge Base"
            }

        ingredients = section_data.get("essential_ingredients", [])
        
        prompt = f"""
        You are an expert Indian Legal Auditor. Perform a Consistency Analysis (IPC-CAM).
        
        GOAL: Check if the FIR NARRATIVE contains factual evidence for each ESSENTIAL INGREDIENT of the applied IPC section.

        FIR NARRATIVE:
        \"\"\"{fir_narrative}\"\"\"

        IPC SECTION: {section_number} - {section_data.get('title')}
        ESSENTIAL INGREDIENTS:
        {json.dumps(ingredients, indent=2)}

        ANALYSIS GUIDELINES:
        1. For each ingredient, determine if it is 'Satisfied', 'Partially Satisfied', or 'Not Found'.
        2. Provide a brief 'Evidence' snippet from the text for satisfied ingredients.
        3. Identify any 'Inconsistencies' or missing legal requirements.
        4. Provide an overall 'Consistency Score' (0.0 to 1.0).

        RETURN ONLY A VALID JSON OBJECT IN THIS FORMAT:
        {{
          "section": "{section_number}",
          "consistency_score": 0.0,
          "overall_assessment": "Summary of alignment",
          "ingredient_analysis": [
            {{
              "ingredient": "The ingredient text",
              "status": "Satisfied / Partially Satisfied / Not Found",
              "evidence": "Supporting text or explanation for why it is missing"
            }}
          ],
          "missing_elements": ["List of requirements not found in facts"],
          "potential_misuse_flag": true/false
        }}
        """

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Analysis failed for Section {section_number}: {e}")
            return {"section": section_number, "status": "error", "reason": str(e)}

    def analyze_full_fir(self, fir_doc: Dict) -> Dict:
        """
        Analyze all applied IPC sections in an FIR document.
        """
        narrative = fir_doc.get("narrative", "")
        sections = fir_doc.get("applied_ipc_sections", [])
        
        logger.info(f"Analyzing FIR {fir_doc.get('fir_number')} with sections {sections}")
        
        results = []
        for sec in sections:
            analysis = self.analyze_consistency(narrative, sec)
            results.append(analysis)
            
        # Overall FIR Assessment
        overall_misuse = any(r.get("potential_misuse_flag", False) for r in results)
        avg_score = sum(r.get("consistency_score", 0) for r in results) / len(results) if results else 0
        
        return {
            "fir_number": fir_doc.get("fir_number"),
            "sections_analyzed": sections,
            "section_results": results,
            "overall_consistency_score": round(avg_score, 2),
            "potential_misuse_detected": overall_misuse
        }

# --- Demo Test ---
if __name__ == "__main__":
    # Example usage (will fail without a valid key)
    # cam = IPCCAMModule(api_key="YOUR_KEY_HERE")
    # result = cam.analyze_consistency("He hit me with a stick.", "323")
    # print(json.dumps(result, indent=2))
    print("IPC-CAM Module Loaded Successfully.")
