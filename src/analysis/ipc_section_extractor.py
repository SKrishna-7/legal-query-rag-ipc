"""
Legal IPC-RAG — IPC Section Extraction Module (Phase 2.5)
=========================================================
Extracts applied IPC sections from unstructured legal text or narratives
when formal headers are missing. Uses advanced heuristics and regex patterns
inspired by Legal NER approaches.
"""

import re
import logging
from typing import List, Set

logger = logging.getLogger(__name__)

class IPCSectionExtractor:
    """
    Advanced extractor for identifying IPC sections in raw text.
    """

    def __init__(self):
        # Comprehensive patterns to catch various legal shorthands for IPC
        self.ipc_context_patterns = [
            r"Indian\s+Penal\s+Code",
            r"\bIPC\b",
            r"\bI\.P\.C\.?\b"
        ]
        
        # Patterns to avoid confusing IPC with CrPC or other acts
        self.exclusion_patterns = [
            r"Cr\.?P\.?C\.?",
            r"Criminal\s+Procedure",
            r"Evidence\s+Act",
            r"Arms\s+Act",
            r"NDPS\s+Act",
            r"POCSO",
            r"UAPA"
        ]

        # Regex to find section numbers (handles compound like 302/34, 120-B, 376(2)(g))
        self.section_regex = r"\b(\d{1,3}[A-Za-z]?)\b"
        
        # Specific trigger phrases
        self.trigger_phrases = [
            r"[Uu]nder\s+[Ss]ections?\s+((?:\d+[A-Za-z]?(?:\s*(?:and|&|,|/|\s)\s*)*)+)\s*(?:of\s+)?(?:the\s+)?(?:IPC|Indian\s+Penal\s+Code)?",
            r"[Uu]/[Ss]\.?\s+((?:\d+[A-Za-z]?(?:\s*(?:and|&|,|/|\s)\s*)*)+)",
            r"[Ss]ections?\s+((?:\d+[A-Za-z]?(?:\s*(?:and|&|,|/|\s)\s*)*)+)\s*(?:of\s+)?(?:the\s+)?(?:IPC|Indian\s+Penal\s+Code)",
            r"((?:\d+[A-Za-z]?(?:\s*(?:and|&|,|/|\s)\s*)*)+)\s*(?:of\s+)?(?:the\s+)?(?:IPC|Indian\s+Penal\s+Code)"
        ]

    def extract_sections(self, text: str) -> List[str]:
        """
        Extracts and normalizes IPC sections from the given text.
        """
        if not text:
            return []

        extracted_sections: Set[str] = set()

        # 1. Look for explicit trigger phrases
        for pattern in self.trigger_phrases:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw_match = match.group(1)
                
                # Check surrounding context for exclusion words (like CrPC)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                is_excluded = any(re.search(ex, context, re.IGNORECASE) for ex in self.exclusion_patterns)
                is_explicitly_ipc = any(re.search(inc, context, re.IGNORECASE) for inc in self.ipc_context_patterns)
                
                # If it explicitly mentions CrPC and NOT IPC, skip it
                if is_excluded and not is_explicitly_ipc:
                    continue
                
                # Extract the actual numbers from the matched phrase
                numbers = re.findall(self.section_regex, raw_match)
                for num in numbers:
                    extracted_sections.add(num.upper())

        # 2. Sort the sections numerically for clean output
        def sort_key(x):
            num_part = re.search(r"\d+", x)
            return int(num_part.group()) if num_part else 0

        return sorted(list(extracted_sections), key=sort_key)

    def extract_from_document(self, document_dict: dict) -> List[str]:
        """
        Extracts sections from a dictionary representing a document (e.g., a Bail Judgment).
        It checks multiple fields.
        """
        sections = []
        
        # Check standard fields first
        if "applied_ipc_sections" in document_dict and document_dict["applied_ipc_sections"]:
            return document_dict["applied_ipc_sections"]
            
        # Fallback to scanning text fields
        text_to_scan = ""
        for field in ["narrative", "facts", "judgment", "raw_text", "summary"]:
            if field in document_dict and document_dict[field]:
                text_to_scan += " " + str(document_dict[field])
                
        if text_to_scan:
            sections = self.extract_sections(text_to_scan)
            
        return sections

if __name__ == "__main__":
    extractor = IPCSectionExtractor()
    
    sample_text = """
    The accused was arrested on Friday. The police have filed a chargesheet 
    under sections 302, 120B and 34 of the Indian Penal Code. The defense argued 
    for bail under section 439 of Cr.P.C. Additionally, an FIR was lodged u/s 420/468 IPC.
    """
    
    print("Testing IPC Section Extractor...")
    print(f"Sample Text:\n{sample_text}")
    sections = extractor.extract_sections(sample_text)
    print(f"Extracted IPC Sections: {sections}")
