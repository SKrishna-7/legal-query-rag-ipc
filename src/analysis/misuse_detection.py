"""
Legal IPC-RAG — Misuse Risk Assessment Engine (Phase 2.6)
=========================================================
Analyzes the output of the IPC-CAM to classify the specific type 
of legal misuse or inconsistency present in the FIR.
"""

from typing import Dict, List

class MisuseDetectionEngine:
    """
    Classifies the type of legal misuse based on IPC-CAM ingredient analysis.
    """
    
    MISUSE_CATEGORIES = {
        "OVERCHARGING": "Applied section is far more severe than the facts support.",
        "MISSING_MENS_REA": "Facts describe an act, but lack evidence of criminal intent/knowledge.",
        "MISSING_ACTUS_REUS": "Facts allege intent/conspiracy, but lack the physical act required.",
        "PROCEDURAL_GAP": "Essential procedural prerequisites (like prior sanctions) are missing.",
        "FACT_LAW_MISMATCH": "The core facts do not align with the definition of the applied section.",
        "NO_MISUSE": "The facts adequately support the applied sections."
    }

    def classify_misuse(self, cam_result: Dict) -> Dict:
        """
        Takes a single section's CAM analysis and returns a misuse classification.
        """
        score = cam_result.get("consistency_score", 1.0)
        ingredient_analysis = cam_result.get("ingredient_analysis", [])
        
        # Default state
        classification = "NO_MISUSE"
        confidence = 1.0
        reasoning = "All essential ingredients appear to be satisfied by the narrative."

        if score >= 0.8:
            return {
                "category": classification,
                "confidence": confidence,
                "reasoning": reasoning
            }

        # Analyze missing ingredients to determine the type of misuse
        missing_ingredients = [
            ing for ing in ingredient_analysis 
            if ing.get("status", "").upper() in ["NOT FOUND", "PARTIALLY SATISFIED"]
        ]

        if not missing_ingredients:
            return {"category": "NO_MISUSE", "confidence": 1.0, "reasoning": reasoning}

        # Heuristic 1: Missing Intent (Mens Rea)
        mens_rea_keywords = ["intent", "knowledge", "dishonestly", "fraudulently", "voluntarily", "reason to believe"]
        missing_intent = any(
            any(kw in ing.get("ingredient", "").lower() for kw in mens_rea_keywords) 
            for ing in missing_ingredients
        )
        
        # Heuristic 2: Missing Physical Act (Actus Reus)
        actus_reus_keywords = ["causes", "does", "makes", "moves", "takes", "commits"]
        missing_act = any(
            any(kw in ing.get("ingredient", "").lower() for kw in actus_reus_keywords) 
            for ing in missing_ingredients
        )

        if missing_intent and not missing_act:
            classification = "MISSING_MENS_REA"
            reasoning = "The physical act occurred, but evidence of criminal intent/knowledge is missing from the facts."
            confidence = 0.85
        elif missing_act and not missing_intent:
            classification = "MISSING_ACTUS_REUS"
            reasoning = "Criminal intent or preparation is alleged, but the required physical act was not completed/described."
            confidence = 0.85
        elif score < 0.4:
            classification = "OVERCHARGING"
            reasoning = "Multiple essential ingredients are missing, suggesting a disproportionately severe section was applied."
            confidence = 0.90
        else:
            classification = "FACT_LAW_MISMATCH"
            reasoning = "Specific elements of the law are not supported by the factual narrative."
            confidence = 0.75

        return {
            "category": classification,
            "description": self.MISUSE_CATEGORIES[classification],
            "confidence": confidence,
            "reasoning": reasoning,
            "missing_elements_count": len(missing_ingredients)
        }

    def process_full_report(self, full_cam_report: Dict) -> Dict:
        """
        Processes the full FIR analysis report and appends misuse classifications.
        """
        for section_result in full_cam_report.get("section_results", []):
            misuse_class = self.classify_misuse(section_result)
            section_result["misuse_classification"] = misuse_class
            
        return full_cam_report

if __name__ == "__main__":
    # Quick Test
    mock_cam = {
        "consistency_score": 0.5,
        "ingredient_analysis": [
            {"ingredient": "The physical act of hitting", "status": "Satisfied"},
            {"ingredient": "Done with the intent to cause grievous hurt", "status": "Not Found"}
        ]
    }
    engine = MisuseDetectionEngine()
    print(engine.classify_misuse(mock_cam))
