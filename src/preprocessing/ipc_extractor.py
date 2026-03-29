"""
Legal IPC-RAG — Legal Section Extraction Module (Phase 2.5)
=========================================================
Extracts applied IPC and BNS (2023) sections from unstructured legal text.
Uses advanced heuristics and regex patterns inspired by Legal NER approaches.
"""

import re
import logging
from typing import List, Set

logger = logging.getLogger(__name__)

class IPCSectionExtractor:
    """
    Advanced extractor for identifying IPC and BNS sections in raw text.
    """

    def __init__(self):
        # Comprehensive patterns to catch various legal shorthands for IPC & BNS
        self.ipc_context_patterns = [
            r"Indian\s+Penal\s+Code",
            r"\bIPC\b",
            r"\bI\.P\.C\.?\b",
            r"Bharatiya\s+Nyaya\s+Sanhita",
            r"\bBNS\b",
            r"\bB\.N\.S\.?\b"
        ]
        
        # Patterns to avoid confusing IPC/BNS with CrPC/BNSS or other acts
        self.exclusion_patterns = [
            r"Cr\.?P\.?C\.?",
            r"Criminal\s+Procedure",
            r"Evidence\s+Act",
            r"Arms\s+Act",
            r"NDPS\s+Act",
            r"POCSO",
            r"UAPA",
            r"BNSS",
            r"BSA"
        ]

        # Regex to find section numbers (handles compound like 302/34, 120-B, and BNS subsections like 318(4))
        self.section_regex = r"\b(\d{1,3}[A-Za-z]?(?:\(\d+[A-Za-z]*\))?)\b"
        
        # Specific trigger phrases updated for BNS
        self.trigger_phrases = [
            r"[Uu]nder\s+[Ss]ections?\s+((?:\d+[A-Za-z]?(?:\(\d+\))?(?:\s*(?:and|&|,|/|\s)\s*)*)+)\s*(?:of\s+)?(?:the\s+)?(?:IPC|Indian\s+Penal\s+Code|BNS|Bharatiya\s+Nyaya\s+Sanhita)?",
            r"[Uu]/[Ss]\.?\s+((?:\d+[A-Za-z]?(?:\(\d+\))?(?:\s*(?:and|&|,|/|\s)\s*)*)+)",
            r"[Ss]ections?\s+((?:\d+[A-Za-z]?(?:\(\d+\))?(?:\s*(?:and|&|,|/|\s)\s*)*)+)\s*(?:of\s+)?(?:the\s+)?(?:IPC|Indian\s+Penal\s+Code|BNS|Bharatiya\s+Nyaya\s+Sanhita)",
            r"((?:\d+[A-Za-z]?(?:\(\d+\))?(?:\s*(?:and|&|,|/|\s)\s*)*)+)\s*(?:of\s+)?(?:the\s+)?(?:IPC|Indian\s+Penal\s+Code|BNS|Bharatiya\s+Nyaya\s+Sanhita)"
        ]

    def extract_sections(self, text: str) -> List[str]:
        """
        Extracts and normalizes IPC/BNS sections from the given text.
        """
        if not text:
            return []

        extracted_sections: Set[str] = set()

        # 1. Look for explicit trigger phrases
        for pattern in self.trigger_phrases:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw_match = match.group(1)
                
                # Check surrounding context for exclusion words
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                is_excluded = any(re.search(ex, context, re.IGNORECASE) for ex in self.exclusion_patterns)
                is_explicitly_included = any(re.search(inc, context, re.IGNORECASE) for inc in self.ipc_context_patterns)
                
                # If it mentions an excluded act and NOT IPC/BNS, skip it
                if is_excluded and not is_explicitly_included:
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

    def extract_mentioned_sections(self, text: str) -> List[str]:
        """Alias for extract_sections to maintain compatibility with app.py"""
        return self.extract_sections(text)

if __name__ == "__main__":
    extractor = IPCSectionExtractor()
    
    sample_text = """
    The accused was arrested on Friday. The police have filed a chargesheet 
    under sections 302, 120B and 34 of the Indian Penal Code. The defense argued 
    for bail under section 439 of Cr.P.C. Additionally, an FIR was lodged u/s 318(4), 61(2) BNS.
    """
    
    print("Testing Section Extractor...")
    print(f"Sample Text:\n{sample_text}")
    sections = extractor.extract_sections(sample_text)
    print(f"Extracted Sections: {sections}")
