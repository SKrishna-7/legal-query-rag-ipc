"""
IPC Section Extraction Module (Phase 2.3)
Extracts IPC sections and factual claims from FIR narratives.
Maps extracted claims to essential ingredients of IPC sections.
"""

import re
import json
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import spacy
except ImportError:
    spacy = None

try:
    import torch
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    torch = None
    SentenceTransformer, util = None, None


class ClaimType(Enum):
    ACTION = "ACTION"
    STATE = "STATE"
    INTENTION = "INTENTION"
    KNOWLEDGE = "KNOWLEDGE"
    RELATIONSHIP = "RELATIONSHIP"

class CertaintyLevel(Enum):
    CERTAIN = "CERTAIN"
    ALLEGED = "ALLEGED"
    UNKNOWN = "UNKNOWN"

@dataclass
class Claim:
    text: str
    subject: str
    predicate: str
    object: str
    claim_type: ClaimType
    temporal: str
    location: str
    certainty: CertaintyLevel

@dataclass
class FactElementMapping:
    section: str
    mappings: Dict[str, List[Dict[str, Any]]]

class IPCSectionExtractor:
    def __init__(self, 
                 ipc_kb_path: str = "data/processed/ipc_sections/ipc_complete.json",
                 embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0"):
        self.ipc_kb_path = ipc_kb_path
        self.ipc_kb = self._load_ipc_kb()
        
        print("Loading spaCy model 'en_core_web_sm'...")
        if spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"Warning: en_core_web_sm not found ({e}). Please install it.")
                self.nlp = None
        else:
            self.nlp = None

        print(f"Loading embedding model '{embedding_model_name}'...")
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
            except Exception as e:
                print(f"Failed to load sentence transformer: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None

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

    def extract_mentioned_sections(self, text: str) -> List[str]:
        """Extract all IPC sections mentioned in text."""
        sections = set()
        
        # Various patterns for IPC section mentions
        patterns = [
            r'[Ss]ection\s+(\d{1,3}[A-Z]?(?:/\d{1,3}[A-Z]?)*)\s*(?:of\s+)?(?:the\s+)?IPC',
            r'[Ss]ection\s+(\d{1,3}[A-Z]?(?:/\d{1,3}[A-Z]?)*)',
            r'[Uu]/[Ss]\.?\s*(\d{1,3}[A-Z]?(?:/\d{1,3}[A-Z]?)*)',
            r'(?:IPC|I\.P\.C\.)\s+(\d{1,3}[A-Z]?(?:/\d{1,3}[A-Z]?)*)',
            r'(\d{1,3}[A-Z]?(?:/\d{1,3}[A-Z]?)*)\s*(?:of\s+)?(?:IPC|I\.P\.C\.)'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                raw_match = match.group(1)
                # Split by slash or 'read with'
                parts = re.split(r'[/,]', raw_match)
                for part in parts:
                    part = part.strip()
                    if re.match(r'^\d{1,3}[A-Z]?$', part):
                        # Validate against KB
                        if part in self.ipc_kb:
                            sections.add(part)
                            
        return sorted(list(sections))

    def extract_facts_as_claims(self, narrative: str) -> List[Claim]:
        """Use spaCy to extract factual claims from FIR narrative."""
        if not self.nlp:
            print("spaCy not loaded. Returning empty claims.")
            return []
            
        doc = self.nlp(narrative)
        claims = []
        
        for sent in doc.sents:
            # Extract basic entities
            temporal = []
            location = []
            for ent in sent.ents:
                if ent.label_ in ["DATE", "TIME"]:
                    temporal.append(ent.text)
                elif ent.label_ in ["GPE", "LOC", "FAC"]:
                    location.append(ent.text)
                    
            # Determine certainty
            text_lower = sent.text.lower()
            if any(word in text_lower for word in ["alleged", "claimed", "accused of", "stated"]):
                certainty = CertaintyLevel.ALLEGED
            elif any(word in text_lower for word in ["perhaps", "maybe", "unknown"]):
                certainty = CertaintyLevel.UNKNOWN
            else:
                certainty = CertaintyLevel.CERTAIN
                
            # Classify claim type (simplified heuristic)
            if any(word in text_lower for word in ["intended", "intention", "planned", "premeditated"]):
                claim_type = ClaimType.INTENTION
            elif any(word in text_lower for word in ["knew", "knowing", "aware"]):
                claim_type = ClaimType.KNOWLEDGE
            elif any(word in text_lower for word in ["husband", "wife", "relative", "friend"]):
                claim_type = ClaimType.RELATIONSHIP
            elif any(word in text_lower for word in ["is", "was", "were", "are", "had", "state"]):
                claim_type = ClaimType.STATE
            else:
                claim_type = ClaimType.ACTION

            # Basic SVO extraction for this sentence
            subj, verb, obj = "", "", ""
            for token in sent:
                if "subj" in token.dep_:
                    subj = " ".join([c.text for c in token.subtree]).strip()
                    verb = token.head.text
                    for child in token.head.children:
                        if "obj" in child.dep_:
                            obj = " ".join([c.text for c in child.subtree]).strip()
                            break
                    break # just grab first main SVO for simplicity

            # Ensure we have at least a verb or substantive text
            if verb or len(sent.text.split()) > 3:
                claims.append(Claim(
                    text=sent.text.strip(),
                    subject=subj,
                    predicate=verb,
                    object=obj,
                    claim_type=claim_type,
                    temporal=", ".join(temporal),
                    location=", ".join(location),
                    certainty=certainty
                ))
                
        return claims

    def map_facts_to_section_elements(self, claims: List[Claim], ipc_sections: List[str]) -> List[FactElementMapping]:
        """Map facts to essential ingredients using sentence embeddings."""
        if not self.embedding_model or not torch:
            print("Embedding model or torch not loaded. Cannot map facts.")
            return []
            
        results = []
        
        # Pre-embed all claims
        claim_texts = [c.text for c in claims]
        if not claim_texts:
            return results
            
        claim_embeddings = self.embedding_model.encode(claim_texts, convert_to_tensor=True)
        
        for section_num in ipc_sections:
            if section_num not in self.ipc_kb:
                print(f"Warning: Section {section_num} not found in KB.")
                continue
                
            section_data = self.ipc_kb[section_num]
            ingredients = section_data.get("essential_ingredients", [])
            
            if not ingredients:
                continue
                
            # Embed ingredients
            ingredient_embeddings = self.embedding_model.encode(ingredients, convert_to_tensor=True)
            
            # Compute cosine similarities (Ingredients x Claims matrix)
            cosine_scores = util.cos_sim(ingredient_embeddings, claim_embeddings)
            
            mapping = FactElementMapping(section=section_num, mappings={})
            
            for i, ingredient in enumerate(ingredients):
                # Find top 3 matching claims for this ingredient
                scores = cosine_scores[i]
                k = min(3, len(claims))
                top_results = torch.topk(scores, k=k)
                
                matched_claims = []
                for score, idx in zip(top_results[0], top_results[1]):
                    if score.item() > 0.3: # Threshold
                        matched_claims.append({
                            "claim": asdict(claims[idx]),
                            "score": float(score.item())
                        })
                
                mapping.mappings[ingredient] = matched_claims
                
            results.append(mapping)
            
        return results

    def save_results(self, extracted_data: Dict[str, Any], output_path: str):
        """Save the extraction results."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            # Need a custom encoder for enums
            class EnumEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Enum):
                        return obj.value
                    return super().default(obj)
                    
            json.dump(extracted_data, f, indent=2, ensure_ascii=False, cls=EnumEncoder)
            
# For standalone testing
if __name__ == "__main__":
    extractor = IPCSectionExtractor()
    sample_text = "On 12/05/2023, the accused deliberately hit the complainant with a metal rod at the local market, causing grievous hurt. The police registered a case under Section 326 of the IPC."
    
    sections = extractor.extract_mentioned_sections(sample_text)
    print("Mentioned sections:", sections)
    
    claims = extractor.extract_facts_as_claims(sample_text)
    print(f"Extracted {len(claims)} claims.")
    
    mappings = extractor.map_facts_to_section_elements(claims, sections)
    
    output = {
        "text": sample_text,
        "sections": sections,
        "claims": [asdict(c) for c in claims],
        "mappings": [{"section": m.section, "mappings": m.mappings} for m in mappings]
    }
    extractor.save_results(output, "data/processed/extraction_test.json")
    print("Test extraction saved to data/processed/extraction_test.json.")
