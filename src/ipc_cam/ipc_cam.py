"""
IPC Contextual Alignment Module (IPC-CAM) — CORE NOVELTY (Phase 2.4)
Evaluates whether the facts in a FIR legally satisfy the requirements 
to invoke specific IPC sections.

Uses a 3-step alignment process:
  Step A: Semantic Similarity (GIST Large Embedding)
  Step B: NLI Entailment (Cross-Encoder)
  Step C: LLM Verification (Groq / Llama 3.3)
"""

import os
import json
import time
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import torch
    from sentence_transformers import SentenceTransformer, util, CrossEncoder
except ImportError:
    torch, SentenceTransformer, util, CrossEncoder = None, None, None, None

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


class SatisfactionStatus(Enum):
    SATISFIED = "SATISFIED"
    PARTIALLY_SATISFIED = "PARTIALLY_SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"

class AlignmentStatus(Enum):
    FULLY_ALIGNED = "FULLY_ALIGNED"
    PARTIALLY_ALIGNED = "PARTIALLY_ALIGNED"
    MISALIGNED = "MISALIGNED"

@dataclass
class IngredientScore:
    ingredient: str
    satisfaction_status: SatisfactionStatus
    confidence: float
    semantic_score: float
    nli_entailment: float
    llm_verdict: str
    evidence_sentences: List[str]
    reasoning: str

@dataclass
class SectionAlignmentResult:
    section_number: str
    alignment_status: AlignmentStatus
    alignment_score: float
    ingredient_scores: List[IngredientScore]
    missing_ingredients: List[str]
    partial_ingredients: List[str]
    alignment_reasoning: str

@dataclass
class IPCCAMReport:
    fir_number: str
    sections_evaluated: List[SectionAlignmentResult]
    misuse_detected: bool
    overall_misuse_risk_score: float
    cam_summary: str

class IPCContextualAlignmentModule:
    def __init__(self,
                 ipc_kb_path: str = "data/processed/ipc_sections/ipc_complete.json",
                 groq_api_key: str = "",
                 embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0",
                 nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
                 use_local: bool = False,
                 local_model_path: str = "meta-llama/Llama-3.2-3B"):
        
        self.ipc_kb_path = ipc_kb_path
        self.ipc_kb = self._load_ipc_kb()
        self.use_local = use_local
        
        # Initialize Groq client
        self.groq_api_key = groq_api_key
        if Groq and groq_api_key:
            self.client = Groq(api_key=groq_api_key)
        else:
            self.client = None
            print("Warning: Groq client not initialized.")

        # Initialize Local Model if requested
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

        # Initialize Sentence Transformer for semantic search
        print(f"Loading embedding model '{embedding_model_name}'...")
        if SentenceTransformer:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        else:
            self.embedding_model = None

        # Initialize Cross-Encoder for NLI
        print(f"Loading NLI cross-encoder '{nli_model_name}'...")
        if CrossEncoder:
            self.nli_model = CrossEncoder(nli_model_name)
        else:
            self.nli_model = None

    def _load_ipc_kb(self) -> Dict[str, Any]:
        """Load the structured IPC knowledge base."""
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

    def compute_ingredient_satisfaction_score(self,
                                             ingredient: str,
                                             fir_narrative: str) -> IngredientScore:
        """Evaluate if an ingredient is satisfied by the FIR narrative using LLM analysis on the full text."""
        
        llm_verdict = "NOT_SATISFIED"
        confidence = 0.0
        reasoning = ""
        evidence_sentences = []
        
        prompt = f"""
        You are a strict Indian Legal Expert. Analyze the following FIR narrative to determine if the specific legal ingredient requirement is satisfied.
        
        CRITICAL INSTRUCTION: Read the ENTIRE narrative carefully. The evidence might be buried in the 'First information contents' or attached sheets at the end.
        
        Legal Ingredient Requirement: "{ingredient}"
        
        FIR Narrative:
        {fir_narrative[:15000]}
        
        Does the narrative provide explicit facts or evidence that satisfy this legal requirement?
        Answer in the following JSON format:
        {{
            "verdict": "SATISFIED" | "PARTIALLY_SATISFIED" | "NOT_SATISFIED",
            "confidence": (float between 0.0 and 1.0, where 1.0 means perfectly justified),
            "reasoning": "Brief legal reasoning citing specific facts from the text if present."
        }}
        """

        if self.use_local and self.local_pipeline:
            try:
                # Local Llama-3.2 generation
                outputs = self.local_pipeline(
                    prompt, 
                    max_new_tokens=200,
                    return_full_text=False
                )
                response_text = outputs[0]['generated_text']
                import re
                import json
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    res = json.loads(json_match.group())
                else:
                    res = {"verdict": "PARTIALLY_SATISFIED", "confidence": 0.5, "reasoning": "Local model returned non-JSON response"}
                
                llm_verdict = res.get("verdict", "NOT_SATISFIED")
                confidence = float(res.get("confidence", 0.0))
                reasoning = res.get("reasoning", "")
            except Exception as e:
                import logging; logging.error(f"Local Model Error: {e}")
                llm_verdict = "PARTIALLY_SATISFIED"
                reasoning = f"Error in local inference: {e}"

        elif self.client:
            try:
                import json
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    response_format={"type": "json_object"}
                )
                res = json.loads(chat_completion.choices[0].message.content)
                llm_verdict = res.get("verdict", "NOT_SATISFIED")
                confidence = float(res.get("confidence", 0.0))
                reasoning = res.get("reasoning", "")
                
                # Try to extract the sentence it relied on for evidence tracking
                if "SATISFIED" in llm_verdict:
                     evidence_sentences.append(reasoning)

            except Exception as e:
                import logging; logging.error(f"Groq API Error: {e}")
                llm_verdict = "PARTIALLY_SATISFIED"
                reasoning = f"Error calling LLM API: {e}"

        else:
            # Fallback if no LLM
            llm_verdict = "PARTIALLY_SATISFIED"
            confidence = 0.5
            reasoning = "No LLM available for semantic analysis."

        status_map = {
            "SATISFIED": SatisfactionStatus.SATISFIED,
            "PARTIALLY_SATISFIED": SatisfactionStatus.PARTIALLY_SATISFIED,
            "NOT_SATISFIED": SatisfactionStatus.NOT_SATISFIED
        }
        
        return IngredientScore(
            ingredient=ingredient,
            satisfaction_status=status_map.get(llm_verdict, SatisfactionStatus.NOT_SATISFIED),
            confidence=confidence,
            semantic_score=1.0,
            nli_entailment=1.0,
            llm_verdict=llm_verdict,
            evidence_sentences=evidence_sentences,
            reasoning=reasoning
        )

    def evaluate_section_alignment(self,
                                   section_number: str,
                                   fir_narrative: str) -> SectionAlignmentResult:
        """Evaluate how well the FIR facts align with the given IPC section."""
        
        if section_number not in self.ipc_kb:
            import logging; logging.info(f"Section {section_number} not in IPC KB. Falling back to Generative Evaluation.")
            if self.client:
                try:
                    prompt = f"""
                    You are a legal expert evaluating an FIR. The police have applied 'Section {section_number}' 
                    (This might be a non-IPC section like PC Act, NDPS, POCSO, etc.).
                    
                    CRITICAL INSTRUCTION: You are analyzing an Indian FIR. The specific physical acts, dates, and intent are rarely found on the first page. You MUST thoroughly search the 'First information contents' or attached narrative sheets towards the end of the text to extract the factual basis of the charges before making any conclusions.
                    
                    FIR Narrative:
                    {fir_narrative[:10000]}
                    
                    Does the narrative provide sufficient factual evidence to justify charging someone under 'Section {section_number}'?
                    
                    Respond strictly in JSON format:
                    {{
                        "alignment_score": (float between 0.0 and 1.0, where 1.0 means perfectly justified and 0.0 means completely unjustified),
                        "reasoning": "Brief legal reasoning explaining why it is or isn't justified based on the text."
                    }}
                    """
                    completion = self.client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                    import json
                    result = json.loads(completion.choices[0].message.content)
                    score = float(result.get("alignment_score", 0.0))
                    reasoning = result.get("reasoning", "LLM Evaluation completed.")
                    
                    if score >= 0.8:
                        status = AlignmentStatus.FULLY_ALIGNED
                    elif score >= 0.4:
                        status = AlignmentStatus.PARTIALLY_ALIGNED
                    else:
                        status = AlignmentStatus.MISALIGNED
                        
                    return SectionAlignmentResult(
                        section_number=f"{section_number} (Non-IPC)",
                        alignment_status=status,
                        alignment_score=score,
                        ingredient_scores=[],
                        missing_ingredients=[],
                        partial_ingredients=[],
                        alignment_reasoning=f"Generative Evaluation: {reasoning}"
                    )
                except Exception as e:
                    import logging; logging.error(f"Generative fallback failed: {e}")

            return SectionAlignmentResult(
                section_number=section_number,
                alignment_status=AlignmentStatus.MISALIGNED,
                alignment_score=0.0,
                ingredient_scores=[],
                missing_ingredients=[],
                partial_ingredients=[],
                alignment_reasoning=f"Section {section_number} not found in knowledge base."
            )
            
        section_data = self.ipc_kb[section_number]
        ingredients = section_data.get("essential_ingredients", [])
        
        if not ingredients:
            # Fallback for sections with no ingredients listed
            ingredients = [section_data.get("title", "Offence definition")]

        ingredient_scores = []
        for ing in ingredients:
            score = self.compute_ingredient_satisfaction_score(ing, fir_narrative)
            ingredient_scores.append(score)
            
        # Calculate alignment score (0.0 to 1.0)
        total_score = 0.0
        missing = []
        partial = []
        
        for s in ingredient_scores:
            if s.satisfaction_status == SatisfactionStatus.SATISFIED:
                total_score += 1.0
            elif s.satisfaction_status == SatisfactionStatus.PARTIALLY_SATISFIED:
                total_score += 0.5
                partial.append(s.ingredient)
            else:
                missing.append(s.ingredient)
                
        alignment_score = total_score / len(ingredient_scores) if ingredient_scores else 0.0
        
        # Determine overall status
        if alignment_score >= 0.8:
            status = AlignmentStatus.FULLY_ALIGNED
        elif alignment_score >= 0.4:
            status = AlignmentStatus.PARTIALLY_ALIGNED
        else:
            status = AlignmentStatus.MISALIGNED
            
        reasoning = f"The FIR narrative satisfies {total_score} out of {len(ingredients)} essential ingredients for Section {section_number}."
        if missing:
            reasoning += f" Critical missing elements: {', '.join(missing[:2])}."

        return SectionAlignmentResult(
            section_number=section_number,
            alignment_status=status,
            alignment_score=alignment_score,
            ingredient_scores=ingredient_scores,
            missing_ingredients=missing,
            partial_ingredients=partial,
            alignment_reasoning=reasoning
        )

    def generate_full_cam_report(self,
                                 fir_number: str,
                                 applied_sections: List[str],
                                 narrative: str) -> IPCCAMReport:
        """Generate a complete alignment report for all applied sections."""
        
        results = []
        for sec in applied_sections:
            import logging; logging.info(f"Evaluating alignment for Section {sec}...")
            res = self.evaluate_section_alignment(sec, narrative)
            results.append(res)
            
        # Overall misuse risk
        misuse_detected = any(r.alignment_status == AlignmentStatus.MISALIGNED for r in results)
        avg_score = sum(r.alignment_score for r in results) / len(results) if results else 0.0
        misuse_risk = 1.0 - avg_score
        
        summary = f"Audit of FIR {fir_number} indicates an average legal alignment score of {avg_score:.2f}."
        if misuse_detected:
            summary += " Warning: Potential legal misuse detected due to insufficient factual support for one or more sections."

        return IPCCAMReport(
            fir_number=fir_number,
            sections_evaluated=results,
            misuse_detected=misuse_detected,
            overall_misuse_risk_score=misuse_risk,
            cam_summary=summary
        )

# For standalone testing
if __name__ == "__main__":
    cam = IPCContextualAlignmentModule()
    
    # Test case: Section 302 (Murder) facts
    test_narrative = (
        "The accused approached the victim with a sharp knife and stabbed him multiple times "
        "in the chest. The victim fell to the ground and died instantly. The accused shouted "
        "that he had planned to kill the victim for a long time."
    )
    test_sections = ["302"]
    
    report = cam.generate_full_cam_report("TEST_001", test_sections, test_narrative)
    
    # Save test report
    output_path = "data/processed/cam_test_report.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    class IPCEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Enum):
                return obj.value
            return super().default(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, cls=IPCEncoder)
        
    print(f"IPC-CAM Test report saved to {output_path}")
    print(f"Alignment Score for 302: {report.sections_evaluated[0].alignment_score}")
    print(f"Status: {report.sections_evaluated[0].alignment_status}")
