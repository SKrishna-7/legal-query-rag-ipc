"""
Legal IPC-RAG — Complete Raw Data Preprocessing Pipeline
==========================================================
Preprocesses ALL downloaded raw data into clean, structured formats
ready for embedding, fine-tuning, and IPC-CAM evaluation.

Handles:
  1. IPC PDF Full Text Extraction & Section Parsing
  2. HuggingFace Dataset Preprocessing (IL-TUR, InLegalNER, Bail Judgments)
  3. GitHub Dataset Preprocessing (Law-AI charge identification)
  4. FIR Document Preprocessing
  5. Case Law JSON Structuring
  6. Embedding Corpus Construction
  7. Training Dataset Assembly
  8. Quality Validation & Stats Report

Run:
  python preprocess.py --all
  python preprocess.py --step ipc_pdf
  python preprocess.py --step huggingface
  python preprocess.py --step fir
  python preprocess.py --step build_corpus
  python preprocess.py --stats
"""

import os
import re
import json
import csv
import glob
import shutil
import argparse
import unicodedata
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Optional imports with graceful fallback ────────────────────────────────
try:
    import fitz          # PyMuPDF
    PYMUPDF_OK = True
except ImportError:
    PYMUPDF_OK = False
    print("WARN: PyMuPDF not installed. PDF parsing disabled.")
    print("      Fix: pip install PyMuPDF")

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    from tqdm import tqdm
    def progress(iterable, desc=""):
        return tqdm(iterable, desc=desc, ncols=80)
except ImportError:
    def progress(iterable, desc=""):
        print(f"  Processing: {desc}")
        return iterable

# ══════════════════════════════════════════════════════════════════════════════
# DIRECTORY CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RAW = {
    "ipc_pdf":      "data/raw/ipc_corpus",           # IPC PDFs
    "fir":          "data/raw/fir_documents",         # FIR PDFs / text files
    "hf":           "data/raw/huggingface",           # HuggingFace downloads
    "github":       "data/raw/github",                # GitHub datasets
    "case_law":     "data/raw/case_law",              # Case law JSONs
    "kaggle":       "data/raw/kaggle",                # Kaggle datasets
}

PROCESSED = {
    "ipc":          "data/processed/ipc_sections",   # Parsed IPC sections
    "fir":          "data/processed/fir_processed",  # Cleaned FIRs
    "corpus":       "data/processed/corpus",         # Embedding corpus chunks
    "train_qa":     "data/processed/train/qa",       # QA training data
    "train_instr":  "data/processed/train/instruction",
    "train_ipc":    "data/processed/train/ipc_specific",
    "synthetic":    "data/processed/synthetic",      # Embedding pairs
    "eval":         "data/evaluation/test_firs",
    "ground_truth": "data/evaluation/ground_truth",
}

LOGS = "logs/preprocessing"

for d in list(RAW.values()) + list(PROCESSED.values()) + [LOGS]:
    Path(d).mkdir(parents=True, exist_ok=True)

LOG_FILE = f"{LOGS}/preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
STATS = defaultdict(int)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_json(data, path: str, desc: str = ""):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    size_kb = Path(path).stat().st_size // 1024
    n = len(data) if isinstance(data, (list, dict)) else 1
    log(f"  Saved: {path} [{n} records | {size_kb} KB] {desc}")
    STATS["files_saved"] += 1
    STATS["total_records"] += n if isinstance(data, list) else 0


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    """Universal text cleaner for legal documents."""
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters except newline/tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Fix common OCR errors
    text = text.replace("|", "I").replace("l", "l")
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page numbers like "Page 12 of 340"
    text = re.sub(r"[Pp]age\s+\d+\s+of\s+\d+", "", text)
    # Remove header/footer artifacts
    text = re.sub(r"─+|═+|_{3,}|={3,}", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 512,
               overlap: int = 50) -> list:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — IPC PDF EXTRACTION & SECTION PARSING
# ══════════════════════════════════════════════════════════════════════════════

# Complete IPC chapter structure
IPC_CHAPTERS = {
    "I":    "Introduction",
    "II":   "General Explanations",
    "III":  "Punishments",
    "IV":   "General Exceptions",
    "V":    "Abetment",
    "VA":   "Criminal Conspiracy",
    "VI":   "Offences Against the State",
    "VII":  "Offences Against the Army",
    "VIII": "Offences Against Public Tranquility",
    "IX":   "Offences by or Relating to Public Servants",
    "X":    "Contempts of Lawful Authority of Public Servants",
    "XI":   "False Evidence and Offences Against Public Justice",
    "XII":  "Offences Relating to Coin and Government Stamps",
    "XIII": "Offences Relating to Weights and Measures",
    "XIV":  "Offences Affecting the Public Health, Safety, Convenience, Decency and Morals",
    "XV":   "Offences Relating to Religion",
    "XVI":  "Offences Affecting the Human Body",
    "XVII": "Offences Against Property",
    "XVIII":"Offences Relating to Documents and Property Marks",
    "XIX":  "Offences Relating to Employment",
    "XX":   "Offences Relating to Marriage",
    "XX-A": "Cruelty by Husband or Relatives of Husband",
    "XXI":  "Defamation",
    "XXII": "Criminal Intimidation, Insult and Annoyance",
    "XXIII":"Attempts to Commit Offences",
}

# Section number to chapter mapping
SECTION_CHAPTER_MAP = {
    **{str(n): "I"    for n in range(1,  6)},
    **{str(n): "II"   for n in range(6,  53)},
    **{str(n): "III"  for n in range(53, 76)},
    **{str(n): "IV"   for n in range(76, 107)},
    **{str(n): "V"    for n in range(107, 120)},
    "120A": "VA", "120B": "VA",
    **{str(n): "VI"   for n in range(121, 131)},
    **{str(n): "VII"  for n in range(131, 141)},
    **{str(n): "VIII" for n in range(141, 161)},
    **{str(n): "IX"   for n in range(161, 172)},
    **{str(n): "X"    for n in range(172, 191)},
    **{str(n): "XI"   for n in range(191, 230)},
    **{str(n): "XII"  for n in range(230, 264)},
    **{str(n): "XIII" for n in range(264, 268)},
    **{str(n): "XIV"  for n in range(268, 295)},
    **{str(n): "XV"   for n in range(295, 299)},
    **{str(n): "XVI"  for n in range(299, 378)},
    "304A": "XVI", "304B": "XVI", "326A": "XVI", "326B": "XVI",
    "354A": "XVI", "354B": "XVI", "354C": "XVI", "354D": "XVI",
    "363A": "XVI", "364A": "XVI", "366A": "XVI", "366B": "XVI",
    "370A": "XVI", "376A": "XVI", "376B": "XVI", "376C": "XVI",
    "376D": "XVI", "376DA": "XVI", "376DB": "XVI", "376E": "XVI",
    **{str(n): "XVII" for n in range(378, 463)},
    **{str(n): "XVIII"for n in range(463, 490)},
    "477A": "XVIII",
    **{str(n): "XX"   for n in range(493, 499)},
    "498A": "XX-A",
    "499": "XXI", "500": "XXI", "501": "XXI", "502": "XXI",
    **{str(n): "XXII" for n in range(503, 511)},
    "511": "XXIII",
}

# Complete section title dictionary
SECTION_TITLES = {
    "1":   "Title and extent of operation of the Code",
    "2":   "Punishment of offences committed within India",
    "3":   "Punishment of offences committed beyond, but which by law may be tried within India",
    "4":   "Extension of Code to extra-territorial offences",
    "34":  "Acts done by several persons in furtherance of common intention",
    "107": "Abetment of a thing",
    "109": "Punishment of abetment if the act abetted is committed in consequence",
    "120A":"Definition of criminal conspiracy",
    "120B":"Punishment of criminal conspiracy",
    "121": "Waging or attempting to wage war or abetting waging of war against the Government of India",
    "141": "Unlawful assembly",
    "149": "Every member of unlawful assembly guilty of offence committed in prosecution of common objects",
    "279": "Rash driving or riding on a public way",
    "295": "Injuring or defiling place of worship with intent to insult the religion of any class",
    "299": "Culpable homicide",
    "300": "Murder",
    "301": "Culpable homicide by causing death of person other than person whose death was intended",
    "302": "Punishment for murder",
    "303": "Punishment for murder by life-convict",
    "304": "Punishment for culpable homicide not amounting to murder",
    "304A":"Causing death by negligence",
    "304B":"Dowry death",
    "305": "Abetment of suicide of child or insane person",
    "306": "Abetment of suicide",
    "307": "Attempt to murder",
    "308": "Attempt to commit culpable homicide",
    "309": "Attempt to commit suicide",
    "319": "Hurt",
    "320": "Grievous hurt",
    "321": "Voluntarily causing hurt",
    "322": "Voluntarily causing grievous hurt",
    "323": "Punishment for voluntarily causing hurt",
    "324": "Voluntarily causing hurt by dangerous weapons or means",
    "325": "Punishment for voluntarily causing grievous hurt",
    "326": "Voluntarily causing grievous hurt by dangerous weapons or means",
    "326A":"Voluntarily causing grievous hurt by use of acid",
    "326B":"Voluntarily throwing or attempting to throw acid",
    "337": "Causing hurt by act endangering life or personal safety of others",
    "338": "Causing grievous hurt by act endangering life or personal safety of others",
    "339": "Wrongful restraint",
    "340": "Wrongful confinement",
    "341": "Punishment for wrongful restraint",
    "342": "Punishment for wrongful confinement",
    "349": "Force",
    "350": "Criminal force",
    "351": "Assault",
    "352": "Punishment for assault or criminal force otherwise than on grave provocation",
    "353": "Assault or criminal force to deter public servant from discharge of his duty",
    "354": "Assault or criminal force to woman with intent to outrage her modesty",
    "354A":"Sexual harassment and punishment for sexual harassment",
    "354B":"Assault or use of criminal force to woman with intent to disrobe",
    "354C":"Voyeurism",
    "354D":"Stalking",
    "359": "Kidnapping",
    "360": "Kidnapping from India",
    "361": "Kidnapping from lawful guardianship",
    "362": "Abduction",
    "363": "Punishment for kidnapping",
    "363A":"Kidnapping or maiming a minor for purposes of begging",
    "364": "Kidnapping or abducting in order to murder",
    "364A":"Kidnapping for ransom",
    "365": "Kidnapping or abducting with intent secretly and wrongfully to confine person",
    "366": "Kidnapping, abducting or inducing woman to compel her marriage",
    "375": "Rape",
    "376": "Punishment for rape",
    "376A":"Punishment for causing death or resulting in persistent vegetative state of victim",
    "376B":"Sexual intercourse by husband upon his wife during separation",
    "376C":"Sexual intercourse by person in authority",
    "376D":"Gang rape",
    "376E":"Punishment for repeat offenders",
    "378": "Theft",
    "379": "Punishment for theft",
    "380": "Theft in dwelling house",
    "381": "Theft by clerk or servant of property in possession of master",
    "382": "Theft after preparation made for causing death, hurt or restraint",
    "383": "Extortion",
    "384": "Punishment for extortion",
    "385": "Putting person in fear of injury in order to commit extortion",
    "386": "Extortion by putting a person in fear of death or grievous hurt",
    "390": "Robbery",
    "391": "Dacoity",
    "392": "Punishment for robbery",
    "393": "Attempt to commit robbery",
    "394": "Voluntarily causing hurt in committing robbery",
    "395": "Punishment for dacoity",
    "396": "Dacoity with murder",
    "397": "Robbery or dacoity with attempt to cause death or grievous hurt",
    "399": "Making preparation to commit dacoity",
    "403": "Dishonest misappropriation of property",
    "404": "Dishonest misappropriation of property possessed by deceased person",
    "405": "Criminal breach of trust",
    "406": "Punishment for criminal breach of trust",
    "407": "Criminal breach of trust by carrier",
    "408": "Criminal breach of trust by clerk or servant",
    "409": "Criminal breach of trust by public servant or banker",
    "415": "Cheating",
    "416": "Cheating by personation",
    "417": "Punishment for cheating",
    "418": "Cheating with knowledge that wrongful loss may ensue",
    "419": "Punishment for cheating by personation",
    "420": "Cheating and dishonestly inducing delivery of property",
    "425": "Mischief",
    "426": "Punishment for mischief",
    "435": "Mischief by fire or explosive substance with intent to cause damage",
    "436": "Mischief by fire or explosive substance with intent to destroy house",
    "441": "Criminal trespass",
    "442": "House-trespass",
    "447": "Punishment for criminal trespass",
    "448": "Punishment for house-trespass",
    "463": "Forgery",
    "464": "Making a false document",
    "465": "Punishment for forgery",
    "466": "Forgery of record of court or of public register",
    "467": "Forgery of valuable security, will, etc.",
    "468": "Forgery for purpose of cheating",
    "469": "Forgery for purpose of harming reputation",
    "471": "Using as genuine a forged document or electronic record",
    "477A":"Falsification of accounts",
    "493": "Cohabitation caused by a man deceitfully inducing a belief of lawful marriage",
    "494": "Marrying again during lifetime of husband or wife",
    "495": "Same offence with concealment of former marriage from person with whom subsequent marriage is contracted",
    "496": "Marriage ceremony fraudulently gone through without lawful marriage",
    "498": "Enticing or taking away or detaining with criminal intent a married woman",
    "498A":"Husband or relative of husband of a woman subjecting her to cruelty",
    "499": "Defamation",
    "500": "Punishment for defamation",
    "503": "Criminal intimidation",
    "504": "Intentional insult with intent to provoke breach of the peace",
    "506": "Punishment for criminal intimidation",
    "507": "Criminal intimidation by an anonymous communication",
    "509": "Word, gesture or act intended to insult the modesty of a woman",
    "511": "Punishment for attempting to commit offences punishable with imprisonment for life or other imprisonment",
}


class IPCPDFParser:
    """
    Parses IPC PDF into structured sections.
    Handles both digital and scanned PDFs.
    """

    # Pattern to detect section headers in PDF text
    SECTION_HEADER_PATTERNS = [
        r"^(\d+[A-Z]?)\.\s+(.+?)[\.\—\-]",          # "302. Punishment for murder."
        r"^[Ss]ection\s+(\d+[A-Z]?)[.\s—]\s*(.+)",  # "Section 302. ..."
        r"^(\d+[A-Z]?)\s*[\.\-—]\s*([A-Z][^.]+)\.",  # "302 - Punishment..."
    ]

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.raw_text = ""
        self.sections = {}

    def extract_text(self) -> str:
        """Extract full text from PDF."""
        if not PYMUPDF_OK:
            log("PyMuPDF not available — trying text file fallback", "WARN")
            txt_path = self.pdf_path.replace(".pdf", ".txt")
            if Path(txt_path).exists():
                with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                    self.raw_text = f.read()
                return self.raw_text
            return ""

        log(f"  Extracting text from: {Path(self.pdf_path).name}")
        doc = fitz.open(self.pdf_path)
        pages_text = []

        for page_num, page in enumerate(progress(doc, "Extracting PDF pages")):
            text = page.get_text("text")

            # If page has very little text, it may be scanned
            if len(text.strip()) < 50:
                try:
                    # Render page as image and OCR
                    import pytesseract
                    from PIL import Image
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img, lang="eng")
                    STATS["ocr_pages"] += 1
                except Exception:
                    pass  # Skip OCR if unavailable

            pages_text.append(text)
            STATS["pages_processed"] += 1

        doc.close()
        self.raw_text = "\n".join(pages_text)
        log(f"  Extracted {len(self.raw_text):,} characters from {len(pages_text)} pages")
        return self.raw_text

    def parse_sections(self) -> dict:
        """Parse raw IPC text into individual sections."""
        if not self.raw_text:
            self.extract_text()

        sections = {}
        text = clean_text(self.raw_text)

        # Strategy 1: Split on section number patterns
        # Look for patterns like "302." or "Section 302" followed by title
        section_pattern = re.compile(
            r'\b(\d{1,3}[A-Z]?)\.\s+([A-Z][a-zA-Z\s,\-]{5,80}?)\.\s*[-—]?\s*'
            r'((?:(?!\b\d{1,3}[A-Z]?\.\s+[A-Z]).)+)',
            re.DOTALL
        )

        matches = list(section_pattern.finditer(text))
        log(f"  Found {len(matches)} section patterns in PDF")

        for i, match in enumerate(matches):
            sec_num = match.group(1).strip()
            title_raw = match.group(2).strip()
            body_raw = match.group(3).strip()

            # Validate section number (must be in known range 1-511)
            try:
                num = int(re.sub(r"[A-Z]", "", sec_num))
                if num < 1 or num > 511:
                    continue
            except ValueError:
                continue

            # Clean
            title = clean_text(title_raw)
            body = clean_text(body_raw)

            if len(body) < 30:
                continue

            chapter_key = SECTION_CHAPTER_MAP.get(sec_num, "UNKNOWN")

            sections[sec_num] = {
                "section_number":       sec_num,
                "title":                SECTION_TITLES.get(sec_num, title),
                "chapter":              chapter_key,
                "chapter_title":        IPC_CHAPTERS.get(chapter_key, ""),
                "full_text":            body,
                "source":               "pdf_parsed",
                "word_count":           len(body.split()),
            }
            STATS["sections_parsed"] += 1

        # Strategy 2: Fallback — use line-by-line parsing
        if len(sections) < 50:
            log("  PDF pattern matching insufficient — switching to line parser", "WARN")
            sections = self._line_based_parser(text)

        self.sections = sections
        log(f"  Successfully parsed {len(sections)} IPC sections from PDF")
        return sections

    def _line_based_parser(self, text: str) -> dict:
        """Fallback parser using line-by-line section detection."""
        sections = {}
        lines = text.split("\n")
        current_section = None
        current_body = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line starts a new section
            match = re.match(r'^(\d{1,3}[A-Z]?)\.\s+([A-Z][a-zA-Z\s,]{5,80})\.',
                             line)
            if match:
                # Save previous section
                if current_section and current_body:
                    body = " ".join(current_body)
                    if len(body) > 30:
                        sections[current_section]["full_text"] = clean_text(body)
                        STATS["sections_parsed"] += 1

                sec_num = match.group(1)
                title = match.group(2).strip()

                try:
                    num = int(re.sub(r"[A-Z]", "", sec_num))
                    if num < 1 or num > 511:
                        continue
                except ValueError:
                    continue

                chapter_key = SECTION_CHAPTER_MAP.get(sec_num, "UNKNOWN")
                current_section = sec_num
                current_body = [line[match.end():].strip()]
                sections[sec_num] = {
                    "section_number": sec_num,
                    "title":          SECTION_TITLES.get(sec_num, title),
                    "chapter":        chapter_key,
                    "chapter_title":  IPC_CHAPTERS.get(chapter_key, ""),
                    "full_text":      "",
                    "source":         "line_parsed",
                    "word_count":     0,
                }
            elif current_section:
                current_body.append(line)

        # Save last section
        if current_section and current_body:
            body = " ".join(current_body)
            if len(body) > 30:
                sections[current_section]["full_text"] = clean_text(body)

        return sections

    def enrich_with_metadata(self, sections: dict) -> dict:
        """Add metadata fields to parsed sections."""
        # Bailable/Non-bailable classification
        NON_BAILABLE = {
            "302","304","304B","307","312","313","314","315","316","320",
            "325","326","326A","326B","354","354B","354C","363","364","364A",
            "365","366","370","370A","375","376","376A","376D","376E",
            "390","391","392","394","395","396","397","399","406","409",
            "420","435","436","449","450","458","459","460","467","468",
            "471","477A","489A","489B","498A","504B","506"
        }
        COGNIZABLE = NON_BAILABLE | {
            "323","324","328","341","342","352","353","357","360","361",
            "362","363A","378","379","380","381","382","383","384","403",
            "405","407","408","415","416","417","418","419","421","422",
            "425","427","428","429","430","431","432","433","434","441",
            "442","447","448","465","466","469","489C","489D"
        }

        for sec_num, section in sections.items():
            section["cognizable"]       = sec_num in COGNIZABLE
            section["bailable"]         = sec_num not in NON_BAILABLE
            section["mens_rea_required"] = self._check_mens_rea(section["full_text"])
            section["keywords"]          = self._extract_keywords(section["full_text"])
            section["punishment_text"]   = self._extract_punishment(section["full_text"])

        return sections

    def _check_mens_rea(self, text: str) -> bool:
        """Detect if section requires mens rea from text."""
        text_lower = text.lower()
        mens_rea_keywords = [
            "intention", "intent", "knowledge", "knowingly",
            "dishonestly", "fraudulently", "wilfully", "deliberately",
            "with intent to"
        ]
        return any(k in text_lower for k in mens_rea_keywords)

    def _extract_keywords(self, text: str) -> list:
        """Extract legal keywords from section text."""
        text_lower = text.lower()
        all_keywords = [
            "murder", "death", "hurt", "grievous", "assault", "rape",
            "theft", "robbery", "extortion", "dacoity", "cheating",
            "forgery", "dowry", "cruelty", "kidnapping", "abduction",
            "defamation", "intimidation", "negligence", "conspiracy",
            "abetment", "dishonest", "fraudulent", "intention", "knowledge",
            "consent", "property", "document", "valuable security",
            "imprisonment", "fine", "death penalty", "life imprisonment"
        ]
        return [k for k in all_keywords if k in text_lower]

    def _extract_punishment(self, text: str) -> str:
        """Extract punishment clause from section text."""
        punishment_pattern = re.compile(
            r'(?:punished|punishment|liable)\s+with\s+(.{20,150}?)[.;]',
            re.IGNORECASE
        )
        match = punishment_pattern.search(text)
        return match.group(0).strip() if match else ""


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — HUGGINGFACE DATASET PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class HuggingFacePreprocessor:
    """Preprocesses downloaded HuggingFace datasets."""

    def process_all(self):
        """Process all HF datasets found in raw/huggingface."""
        log("\n" + "="*60)
        log("MODULE 2: HuggingFace Dataset Preprocessing")
        log("="*60)

        self._process_il_tur()
        self._process_inlegal_ner()
        self._process_bail_judgments()
        self._process_indian_legal()
        self._process_indian_penal_code_hf()

    def _process_il_tur(self):
        """IL-TUR Benchmark — 66k docs + 100 IPC statutes."""
        log("\n[HF-1] Processing IL-TUR Dataset...")

        # IL-TUR is downloaded as parquet files
        parquet_files = glob.glob(f"{RAW['hf']}/IL-TUR/**/*.parquet", recursive=True)
        json_files    = glob.glob(f"{RAW['hf']}/IL-TUR/**/*.json*", recursive=True)
        csv_files     = glob.glob(f"{RAW['hf']}/IL-TUR/**/*.csv", recursive=True)

        all_files = parquet_files + json_files + csv_files

        if not all_files:
            log("  IL-TUR files not found. Expected at data/raw/huggingface/IL-TUR/", "WARN")
            log("  Download: huggingface-cli download Exploration-Lab/IL-TUR", "WARN")
            return

        records = []
        for fpath in progress(all_files, "IL-TUR"):
            try:
                if fpath.endswith(".parquet") and PANDAS_OK:
                    df = pd.read_parquet(fpath)
                    for _, row in df.iterrows():
                        records.append(self._normalize_iltur_row(row.to_dict()))
                elif fpath.endswith(".json") or fpath.endswith(".jsonl"):
                    records.extend(self._load_json_or_jsonl(fpath))
                elif fpath.endswith(".csv") and PANDAS_OK:
                    df = pd.read_csv(fpath)
                    for _, row in df.iterrows():
                        records.append(self._normalize_iltur_row(row.to_dict()))
            except Exception as e:
                log(f"  Skip {Path(fpath).name}: {e}", "WARN")

        if records:
            save_json(records, f"{PROCESSED['train_qa']}/il_tur_processed.json")
            self._extract_ipc_statute_pairs(records)

    def _normalize_iltur_row(self, row: dict) -> dict:
        """Normalize IL-TUR row to standard format."""
        text = str(row.get("text", row.get("document", row.get("content", ""))))
        label = str(row.get("label", row.get("category", row.get("task", ""))))
        question = str(row.get("query", row.get("question", "")))
        answer = str(row.get("answer", row.get("response", "")))
        return {
            "text":     clean_text(text),
            "label":    label,
            "question": clean_text(question),
            "answer":   clean_text(answer),
            "source":   "il_tur"
        }

    def _extract_ipc_statute_pairs(self, records: list):
        """Extract IPC section → statute explanation pairs from IL-TUR."""
        ipc_pattern = re.compile(r'\b(\d{1,3}[A-Z]?)\s+IPC\b|'
                                  r'[Ss]ection\s+(\d{1,3}[A-Z]?)\s+IPC')
        pairs = []
        for rec in records:
            text = rec.get("text", "") + " " + rec.get("answer", "")
            matches = ipc_pattern.findall(text)
            sections = [m[0] or m[1] for m in matches if m[0] or m[1]]
            if sections and len(text) > 100:
                pairs.append({
                    "ipc_sections": list(set(sections)),
                    "text":         text[:2000],
                    "type":         "statute_pair",
                    "source":       "il_tur_extracted"
                })

        if pairs:
            save_json(pairs, f"{PROCESSED['train_ipc']}/iltur_ipc_pairs.json")

    def _process_inlegal_ner(self):
        """InLegalNER — NER annotations for Indian legal text."""
        log("\n[HF-2] Processing InLegalNER Dataset...")
        files = glob.glob(f"{RAW['hf']}/InLegalNER/**/*", recursive=True)
        files = [f for f in files if Path(f).is_file() and
                 Path(f).suffix in [".json", ".jsonl", ".conll", ".txt", ".parquet"]]

        if not files:
            log("  InLegalNER not found. Download: huggingface-cli download opennyaiorg/InLegalNER", "WARN")
            return

        records = []
        for fpath in progress(files, "InLegalNER"):
            try:
                if fpath.endswith(".jsonl") or fpath.endswith(".json"):
                    records.extend(self._load_json_or_jsonl(fpath))
                elif fpath.endswith(".parquet") and PANDAS_OK:
                    df = pd.read_parquet(fpath)
                    records.extend(df.to_dict(orient="records"))
                elif fpath.endswith(".conll") or fpath.endswith(".txt"):
                    records.extend(self._parse_conll(fpath))
            except Exception as e:
                log(f"  Skip {Path(fpath).name}: {e}", "WARN")

        if records:
            # Extract sentences with legal entity labels for training
            entity_records = []
            for rec in records:
                text = str(rec.get("text", rec.get("sentence", "")))
                entities = rec.get("entities", rec.get("ner_tags", []))
                if text and len(text) > 20:
                    entity_records.append({
                        "text":     clean_text(text),
                        "entities": entities,
                        "source":   "inlegal_ner"
                    })

            save_json(entity_records,
                      f"{PROCESSED['train_ipc']}/inlegal_ner_processed.json")

    def _process_bail_judgments(self):
        """Indian Bail Judgments — 1200 bail orders with IPC sections."""
        log("\n[HF-3] Processing Indian Bail Judgments Dataset...")

        search_dirs = [
            f"{RAW['hf']}/IndianBailJudgments-1200",
            f"{RAW['hf']}/IndianBailJudgments",
            f"{RAW['github']}/IndianBailJudgments-1200",
        ]

        all_files = []
        for d in search_dirs:
            all_files.extend(glob.glob(f"{d}/**/*", recursive=True))
        all_files = [f for f in all_files if Path(f).is_file()
                     and Path(f).suffix in [".json", ".jsonl", ".csv", ".txt", ".parquet"]]

        if not all_files:
            log("  Bail Judgments not found.", "WARN")
            log("  Download: huggingface-cli download SnehaDeshmukh/IndianBailJudgments-1200", "WARN")
            return

        records = []
        for fpath in progress(all_files, "Bail Judgments"):
            try:
                if fpath.endswith(".json") or fpath.endswith(".jsonl"):
                    records.extend(self._load_json_or_jsonl(fpath))
                elif fpath.endswith(".csv") and PANDAS_OK:
                    df = pd.read_csv(fpath, encoding="utf-8", errors="replace")
                    records.extend(df.to_dict(orient="records"))
                elif fpath.endswith(".parquet") and PANDAS_OK:
                    df = pd.read_parquet(fpath)
                    records.extend(df.to_dict(orient="records"))
                elif fpath.endswith(".txt"):
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    records.append({"text": text, "source": "bail_txt"})
            except Exception as e:
                log(f"  Skip {Path(fpath).name}: {e}", "WARN")

        if records:
            processed = []
            for rec in records:
                text = str(rec.get("text", rec.get("judgment", rec.get("content", rec.get("facts", "")))))
                if "ipc_sections" in rec and isinstance(rec["ipc_sections"], list):
                    ipc_sections = [str(s) for s in rec["ipc_sections"]]
                else:
                    ipc_sections = self._extract_ipc_sections_from_text(text)
                outcome = str(rec.get("label", rec.get("outcome",
                                      rec.get("bail_granted", rec.get("bail_outcome", "")))))
                if text and len(text) > 100:
                    processed.append({
                        "text":            clean_text(text[:3000]),
                        "ipc_sections":    ipc_sections,
                        "outcome":         outcome,
                        "word_count":      len(text.split()),
                        "source":          "bail_judgments"
                    })

            save_json(processed,
                      f"{PROCESSED['train_ipc']}/bail_judgments_processed.json")

            # Create QA pairs from bail judgments
            qa_pairs = self._generate_bail_qa_pairs(processed)
            save_json(qa_pairs,
                      f"{PROCESSED['train_qa']}/bail_judgment_qa.json")

    def _generate_bail_qa_pairs(self, records: list) -> list:
        """Generate Q&A training pairs from bail judgment records."""
        qa = []
        for rec in records:
            if not rec.get("ipc_sections"):
                continue
            sections_str = ", ".join(
                [f"Section {s} IPC" for s in rec["ipc_sections"][:3]])
            outcome = rec.get("outcome", "")

            qa.append({
                "question": (f"What IPC sections were involved in this case and "
                             f"what was the bail outcome?"),
                "context":  rec["text"][:1000],
                "answer":   (f"The case involved {sections_str}. "
                             f"Bail outcome: {outcome}."),
                "task":     "bail_judgment_qa",
                "source":   "bail_judgments"
            })

        return qa

    def _process_indian_legal(self):
        """Indian Legal text corpus."""
        log("\n[HF-4] Processing Indian Legal Dataset...")
        files = (
            glob.glob(f"{RAW['hf']}/indian-legal/**/*", recursive=True) +
            glob.glob(f"{RAW['hf']}/indian_legal/**/*", recursive=True)
        )
        files = [f for f in files if Path(f).is_file()
                 and Path(f).suffix in [".json", ".jsonl", ".txt", ".parquet"]]

        if not files:
            log("  Indian Legal dataset not found.", "WARN")
            return

        all_texts = []
        for fpath in progress(files, "Indian Legal"):
            try:
                if fpath.endswith(".txt"):
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        all_texts.append(clean_text(f.read()))
                elif fpath.endswith(".jsonl") or fpath.endswith(".json"):
                    data = self._load_json_or_jsonl(fpath)
                    for item in data:
                        text = str(item.get("text", item.get("content", "")))
                        if text:
                            all_texts.append(clean_text(text))
                elif fpath.endswith(".parquet") and PANDAS_OK:
                    df = pd.read_parquet(fpath)
                    for col in ["text", "content", "document"]:
                        if col in df.columns:
                            all_texts.extend(df[col].dropna().astype(str).tolist())
                            break
            except Exception as e:
                log(f"  Skip {Path(fpath).name}: {e}", "WARN")

        if all_texts:
            # Build chunked corpus for embedding
            corpus_chunks = []
            for text in all_texts:
                for chunk in chunk_text(text, chunk_size=512, overlap=50):
                    if len(chunk.split()) > 30:
                        corpus_chunks.append({
                            "text":   chunk,
                            "source": "indian_legal"
                        })

            save_json(corpus_chunks,
                      f"{PROCESSED['corpus']}/indian_legal_chunks.json")

    def _process_indian_penal_code_hf(self):
        """harshitv804/Indian_Penal_Code HuggingFace dataset."""
        log("\n[HF-5] Processing Indian Penal Code HF Dataset...")

        search_paths = [
            f"{RAW['hf']}/Indian_Penal_Code",
            f"{RAW['hf']}/Indian-Penal-Code",
            f"{RAW['hf']}/indian_penal_code",
        ]

        files = []
        for sp in search_paths:
            files.extend(glob.glob(f"{sp}/**/*", recursive=True))
        files = [f for f in files if Path(f).is_file()
                 and Path(f).suffix in [".json", ".jsonl", ".csv", ".parquet", ".txt"]]

        if not files:
            log("  IPC HF Dataset not found.", "WARN")
            log("  Download: huggingface-cli download harshitv804/Indian_Penal_Code", "WARN")
            return

        records = []
        for fpath in progress(files, "IPC HF Dataset"):
            try:
                if fpath.endswith(".json") or fpath.endswith(".jsonl"):
                    records.extend(self._load_json_or_jsonl(fpath))
                elif fpath.endswith(".csv") and PANDAS_OK:
                    df = pd.read_csv(fpath, encoding="utf-8", errors="replace")
                    records.extend(df.to_dict(orient="records"))
                elif fpath.endswith(".parquet") and PANDAS_OK:
                    df = pd.read_parquet(fpath)
                    records.extend(df.to_dict(orient="records"))
            except Exception as e:
                log(f"  Skip {Path(fpath).name}: {e}", "WARN")

        if records:
            # Normalize to IPC section format
            ipc_records = []
            for rec in records:
                sec = str(rec.get("section", rec.get("Section",
                          rec.get("section_number", ""))))
                title = str(rec.get("title", rec.get("Title",
                            rec.get("section_title", ""))))
                text = str(rec.get("text", rec.get("description",
                           rec.get("content", rec.get("body", "")))))
                if sec and text:
                    ipc_records.append({
                        "section_number": sec.strip(),
                        "title":          clean_text(title),
                        "full_text":      clean_text(text),
                        "source":         "hf_ipc_dataset"
                    })

            save_json(ipc_records,
                      f"{PROCESSED['ipc']}/hf_ipc_sections.json")
            log(f"  Extracted {len(ipc_records)} IPC section records")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_json_or_jsonl(self, path: str) -> list:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
        if not content:
            return []
        # Try JSONL first
        if content.startswith("{"):
            records = []
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if records:
                return records
        # Try regular JSON
        try:
            data = json.loads(content)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            return []

    def _parse_conll(self, path: str) -> list:
        """Parse CoNLL format annotation files."""
        records = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        current_tokens = []
        current_tags = []
        for line in lines:
            line = line.strip()
            if not line:
                if current_tokens:
                    records.append({
                        "text": " ".join(current_tokens),
                        "ner_tags": current_tags,
                        "source": "conll"
                    })
                    current_tokens = []
                    current_tags = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current_tokens.append(parts[0])
                    current_tags.append(parts[-1])
        return records

    def _extract_ipc_sections_from_text(self, text: str) -> list:
        """Extract IPC section numbers from text."""
        pattern = re.compile(
            r'[Ss]ection\s+(\d{1,3}[A-Z]?)\s+(?:of\s+)?(?:the\s+)?IPC|'
            r'[Uu]/[Ss]\s+(\d{1,3}[A-Z]?)|'
            r'(\d{1,3}[A-Z]?)\s+IPC',
            re.IGNORECASE
        )
        matches = pattern.findall(text)
        sections = set()
        for match in matches:
            sec = match[0] or match[1] or match[2]
            if sec:
                try:
                    num = int(re.sub(r"[A-Z]", "", sec))
                    if 1 <= num <= 511:
                        sections.add(sec)
                except ValueError:
                    pass
        return sorted(list(sections))


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — GITHUB DATASET PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class GitHubDatasetPreprocessor:
    """Preprocesses datasets downloaded from GitHub repos."""

    def process_all(self):
        log("\n" + "="*60)
        log("MODULE 3: GitHub Dataset Preprocessing")
        log("="*60)
        self._process_auto_charge_identification()
        self._process_kaggle_ipc()

    def _process_auto_charge_identification(self):
        """
        Law-AI/automatic-charge-identification
        FIR facts → IPC charge mapping — most directly useful for IPC-CAM
        """
        log("\n[GH-1] Processing Auto Charge Identification Dataset...")

        search_paths = [
            f"{RAW['github']}/automatic-charge-identification",
            f"{RAW['github']}/Law-AI",
            f"{RAW['hf']}/automatic-charge-identification",
        ]

        all_files = []
        for sp in search_paths:
            all_files.extend(glob.glob(f"{sp}/**/*", recursive=True))
        all_files = [f for f in all_files if Path(f).is_file()
                     and Path(f).suffix in [".json", ".jsonl", ".csv", ".txt"]]

        if not all_files:
            log("  Auto Charge ID dataset not found.", "WARN")
            log("  Download: https://github.com/Law-AI/automatic-charge-identification", "WARN")
            return

        records = []
        for fpath in progress(all_files, "Charge Identification"):
            try:
                if fpath.endswith(".json") or fpath.endswith(".jsonl"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    for line in content.strip().split("\n"):
                        if line.strip():
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                elif fpath.endswith(".csv") and PANDAS_OK:
                    df = pd.read_csv(fpath, encoding="utf-8", errors="replace")
                    records.extend(df.to_dict(orient="records"))
                elif fpath.endswith(".txt"):
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                records.append({"text": line})
            except Exception as e:
                log(f"  Skip {Path(fpath).name}: {e}", "WARN")

        if not records:
            log("  No records extracted from auto-charge-id files", "WARN")
            return

        # Normalize to standard format
        normalized = []
        for rec in records:
            facts = str(rec.get("facts", rec.get("text",
                        rec.get("fir_text", rec.get("narrative", "")))))
            charges = rec.get("charges", rec.get("sections",
                      rec.get("ipc_sections", rec.get("label", 
                      rec.get("chargeid", rec.get("doc_labels", []))))))

            if isinstance(charges, list):
                charges_list = [str(c) for c in charges]
            elif isinstance(charges, str):
                charges_list = [c.strip() for c in
                                re.split(r"[,;/\s]+", charges) if c.strip()]
            else:
                charges_list = []

            # Map common string labels to IPC sections
            label_map = {
                "murder": "302",
                "criminal conspiracy": "120B",
                "hurt": "323",
                "forgery": "463",
                "marriage offence": "498A",
                "offence related to religion": "295A",
                "offence against public justice": "191",
                "theft": "378",
                "robbery": "392",
                "kidnapping": "363",
                "extortion": "383",
                "cheating": "415",
                "mischief": "425",
                "criminal trespass": "441",
                "defamation": "499",
                "criminal intimidation": "503"
            }

            # Extract section numbers
            ipc_sections = []
            for c in charges_list:
                c_lower = c.lower().replace("_", " ").strip()
                if c_lower in label_map:
                    ipc_sections.append(label_map[c_lower])
                else:
                    nums = re.findall(r'\d{1,3}[A-Z]?', c)
                    ipc_sections.extend(nums)

            if facts and len(facts) > 30:
                normalized.append({
                    "facts":          clean_text(facts),
                    "ipc_sections":   list(set(ipc_sections)),
                    "original_label": str(charges),
                    "word_count":     len(facts.split()),
                    "source":         "auto_charge_identification"
                })

        if normalized:
            save_json(normalized,
                      f"{PROCESSED['train_ipc']}/charge_identification.json")

            # Create IPC-CAM training pairs
            cam_pairs = self._create_ipc_cam_training_pairs(normalized)
            save_json(cam_pairs,
                      f"{PROCESSED['train_ipc']}/ipc_cam_training_pairs.json")

    def _create_ipc_cam_training_pairs(self, records: list) -> list:
        """
        Create training pairs for IPC-CAM from charge identification data.
        Format: {facts, section, alignment_label, ingredient_satisfaction}
        """
        pairs = []
        for rec in records:
            facts = rec["facts"]
            sections = rec["ipc_sections"]

            for section in sections[:3]:  # Max 3 sections per record
                pairs.append({
                    "narrative":       facts[:1000],
                    "section":         section,
                    "alignment_label": "ALIGNED",  # Ground truth — these are correct charges
                    "source":          "charge_id_positive"
                })

        log(f"  Created {len(pairs)} IPC-CAM training pairs")
        return pairs

    def _process_kaggle_ipc(self):
        """Kaggle IPC Sections Information dataset."""
        log("\n[GH-2] Processing Kaggle IPC Dataset...")

        kaggle_paths = [
            f"{RAW['kaggle']}/indian-penal-code-ipc-sections-information",
            f"{RAW['kaggle']}/ipc_sections",
            f"{RAW['hf']}/kaggle_ipc",
        ]

        files = []
        for kp in kaggle_paths:
            files.extend(glob.glob(f"{kp}/**/*.csv", recursive=True))
            files.extend(glob.glob(f"{kp}/**/*.json", recursive=True))

        if not files:
            log("  Kaggle IPC dataset not found.", "WARN")
            log("  Download: https://www.kaggle.com/datasets/dev523/indian-penal-code-ipc-sections-information", "WARN")
            return

        records = []
        for fpath in files:
            try:
                if fpath.endswith(".csv") and PANDAS_OK:
                    df = pd.read_csv(fpath, encoding="utf-8", errors="replace")
                    log(f"  Columns found: {list(df.columns)}")
                    for _, row in df.iterrows():
                        rec = row.to_dict()
                        # Try common column names
                        section = str(rec.get("Section", rec.get("section",
                                     rec.get("IPC_Section", ""))))
                        description = str(rec.get("Description", rec.get("description",
                                          rec.get("Offense", rec.get("offense", "")))))
                        punishment = str(rec.get("Punishment", rec.get("punishment", "")))
                        if section and description:
                            records.append({
                                "section_number": section.strip(),
                                "description":    clean_text(description),
                                "punishment":     clean_text(punishment),
                                "cognizable":     str(rec.get("Cognizable", "")),
                                "bailable":       str(rec.get("Bailable", "")),
                                "triable_by":     str(rec.get("Triable", rec.get("Court", ""))),
                                "source":         "kaggle_ipc"
                            })
            except Exception as e:
                log(f"  Skip {Path(fpath).name}: {e}", "WARN")

        if records:
            save_json(records, f"{PROCESSED['ipc']}/kaggle_ipc_sections.json")
            log(f"  Processed {len(records)} IPC sections from Kaggle")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — FIR DOCUMENT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class FIRDataPreprocessor:
    """Preprocesses raw FIR documents."""

    IPC_SECTION_PATTERN = re.compile(
        r'[Ss]ection\s+(\d{1,3}[A-Z]?(?:/\d{1,3}[A-Z]?)*)\s*'
        r'(?:r/?w\.?\s*\d{1,3}[A-Z]?\s*)?(?:of\s+)?(?:the\s+)?IPC|'
        r'[Uu]/[Ss]\s+(\d{1,3}[A-Z]?(?:/\d{1,3}[A-Z]?)*)|'
        r'(\d{1,3}[A-Z]?)\s+(?:of\s+)?I\.?P\.?C\.?',
        re.IGNORECASE
    )

    NARRATIVE_STARTS = [
        "facts of the case", "brief facts", "statement of complainant",
        "details of complaint", "complaint details", "facts alleged",
        "particulars of offence", "it is alleged", "the complainant states",
        "description of offence", "the informant states", "brief facts of case"
    ]

    NARRATIVE_ENDS = [
        "ipc section", "sections applied", "section applied", "offence",
        "officer in charge", "investigated by", "arrested", "signature of"
    ]

    def process_all(self):
        log("\n" + "="*60)
        log("MODULE 4: FIR Document Preprocessing")
        log("="*60)

        all_firs = []

        # Process PDF FIRs
        pdf_files = glob.glob(f"{RAW['fir']}/**/*.pdf", recursive=True)
        if pdf_files:
            log(f"  Found {len(pdf_files)} FIR PDFs")
            for pdf in progress(pdf_files, "FIR PDFs"):
                fir = self._process_fir_pdf(pdf)
                if fir:
                    all_firs.append(fir)

        # Process text FIRs
        txt_files = glob.glob(f"{RAW['fir']}/**/*.txt", recursive=True)
        if txt_files:
            log(f"  Found {len(txt_files)} FIR text files")
            for txt in progress(txt_files, "FIR TXTs"):
                fir = self._process_fir_text(txt)
                if fir:
                    all_firs.append(fir)

        # Process JSON FIRs
        json_files = glob.glob(f"{RAW['fir']}/**/*.json", recursive=True)
        if json_files:
            for jf in json_files:
                try:
                    data = load_json(jf)
                    if isinstance(data, list):
                        all_firs.extend([self._normalize_fir(f) for f in data])
                    elif isinstance(data, dict):
                        all_firs.append(self._normalize_fir(data))
                except Exception as e:
                    log(f"  Skip {Path(jf).name}: {e}", "WARN")

        if not all_firs:
            log("  No FIR files found. Add FIR documents to data/raw/fir_documents/", "WARN")
            log("  Supported formats: PDF, TXT, JSON", "WARN")
            return

        # Deduplicate
        seen = set()
        unique_firs = []
        for fir in all_firs:
            key = fir.get("fir_number", "") + fir.get("narrative", "")[:50]
            if key not in seen:
                seen.add(key)
                unique_firs.append(fir)

        # Split into train/eval
        split_idx = int(len(unique_firs) * 0.8)
        train_firs = unique_firs[:split_idx]
        eval_firs  = unique_firs[split_idx:]

        save_json(train_firs, f"{PROCESSED['fir']}/fir_train.json")
        save_json(eval_firs,  f"{PROCESSED['eval']}/fir_eval.json")
        save_json(unique_firs, f"{PROCESSED['fir']}/fir_all.json")
        log(f"  Total FIRs: {len(unique_firs)} | Train: {len(train_firs)} | Eval: {len(eval_firs)}")

    def _process_fir_pdf(self, pdf_path: str) -> dict:
        if not PYMUPDF_OK:
            return None
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join([p.get_text("text") for p in doc])
            doc.close()
            fir = self._parse_fir_text(clean_text(text))
            fir["source_file"] = Path(pdf_path).name
            fir["source_type"] = "pdf"
            return fir
        except Exception as e:
            log(f"  FIR PDF error {Path(pdf_path).name}: {e}", "WARN")
            return None

    def _process_fir_text(self, txt_path: str) -> dict:
        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            fir = self._parse_fir_text(clean_text(text))
            fir["source_file"] = Path(txt_path).name
            fir["source_type"] = "text"
            return fir
        except Exception as e:
            log(f"  FIR text error {Path(txt_path).name}: {e}", "WARN")
            return None

    def _parse_fir_text(self, text: str) -> dict:
        """Parse FIR text into structured fields."""
        fir = {
            "fir_number":         self._extract_fir_number(text),
            "police_station":     self._extract_police_station(text),
            "district":           self._extract_field(text, ["district", "जिला"]),
            "state":              self._extract_field(text, ["state", "राज्य"]),
            "date_of_report":     self._extract_date(text, "report"),
            "date_of_incident":   self._extract_date(text, "incident"),
            "complainant":        self._extract_complainant(text),
            "accused":            self._extract_accused(text),
            "place_of_occurrence":self._extract_place(text),
            "narrative":          self._extract_narrative(text),
            "applied_ipc_sections":self._extract_ipc_sections(text),
            "raw_text_length":    len(text),
            "word_count":         len(text.split()),
        }
        fir["is_valid"] = bool(fir["narrative"] and
                               len(fir["narrative"].split()) >= 20)
        return fir

    def _normalize_fir(self, data: dict) -> dict:
        """Normalize a dict-format FIR to standard structure."""
        return {
            "fir_number":         str(data.get("fir_number", data.get("FIR_No", ""))),
            "police_station":     str(data.get("police_station", data.get("PS", ""))),
            "district":           str(data.get("district", "")),
            "state":              str(data.get("state", "")),
            "date_of_incident":   str(data.get("date_of_incident", data.get("incident_date", ""))),
            "complainant":        str(data.get("complainant", data.get("informant", ""))),
            "accused":            data.get("accused", []),
            "narrative":          clean_text(str(data.get("narrative",
                                             data.get("facts", data.get("text", ""))))),
            "applied_ipc_sections": data.get("applied_ipc_sections",
                                             data.get("ipc_sections", [])),
            "is_valid":           True,
            "source_type":        "json"
        }

    def _extract_fir_number(self, text: str) -> str:
        patterns = [
            r'FIR\s*[Nn]o\.?\s*:?\s*(\d+[\s/]\d+)',
            r'Crime\s*[Nn]o\.?\s*:?\s*(\d+/\d+)',
            r'Case\s*[Nn]o\.?\s*:?\s*(\d+/\d+)',
            r'[Rr]egistration\s*[Nn]o\.?\s*:?\s*(\d+)',
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()
        return ""

    def _extract_police_station(self, text: str) -> str:
        patterns = [
            r'[Pp]olice\s+[Ss]tation\s*:?\s*([A-Za-z\s]+?)(?:\n|,|\.|District)',
            r'P\.?S\.?\s*:?\s*([A-Za-z\s]+?)(?:\n|,)',
            r'[Tt]hana\s*:?\s*([A-Za-z\s]+?)(?:\n|,)',
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()[:50]
        return ""

    def _extract_field(self, text: str, markers: list) -> str:
        for marker in markers:
            pattern = rf'{marker}\s*:?\s*([A-Za-z\s]+?)(?:\n|,|\.|$)'
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:50]
        return ""

    def _extract_date(self, text: str, date_type: str) -> str:
        if date_type == "report":
            markers = ["date of report", "date:", "fir date", "registered on"]
        else:
            markers = ["date of incident", "date of occurrence",
                       "incident date", "on", "occurred on"]

        date_patterns = [
            r'(\d{2}[/-]\d{2}[/-]\d{4})',
            r'(\d{1,2}\s+\w+\s+\d{4})',
            r'(\w+\s+\d{1,2},?\s+\d{4})',
        ]

        for marker in markers:
            idx = text.lower().find(marker.lower())
            if idx >= 0:
                snippet = text[idx:idx + 80]
                for dp in date_patterns:
                    m = re.search(dp, snippet)
                    if m:
                        return m.group(1)

        # Global date extraction as fallback
        for dp in date_patterns:
            m = re.search(dp, text)
            if m:
                return m.group(1)
        return ""

    def _extract_complainant(self, text: str) -> str:
        patterns = [
            r'[Cc]omplainant\s*:?\s*(?:[Nn]ame\s*:?\s*)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'[Ii]nformant\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'[Rr]eported\s+by\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()
        return ""

    def _extract_accused(self, text: str) -> list:
        patterns = [
            r'[Aa]ccused\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'[Ss]uspect\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]
        accused = []
        for p in patterns:
            for m in re.finditer(p, text):
                name = m.group(1).strip()
                if name not in accused:
                    accused.append(name)
        return accused[:5]

    def _extract_place(self, text: str) -> str:
        patterns = [
            r'[Pp]lace\s+of\s+[Oo]ccurrence\s*:?\s*([^\n.]{5,80})',
            r'[Ll]ocation\s*:?\s*([^\n.]{5,80})',
            r'[Ss]cene\s+of\s+(?:the\s+)?[Cc]rime\s*:?\s*([^\n.]{5,80})',
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()
        return ""

    def _extract_narrative(self, text: str) -> str:
        text_lower = text.lower()
        start_idx = -1
        for marker in self.NARRATIVE_STARTS:
            idx = text_lower.find(marker)
            if idx >= 0:
                start_idx = idx + len(marker)
                while start_idx < len(text) and text[start_idx] in ":\n\r\t ":
                    start_idx += 1
                break

        if start_idx < 0:
            lines = [l for l in text.split("\n") if len(l.split()) > 5]
            start_idx = text.find(lines[2]) if len(lines) > 2 else 0

        end_idx = len(text)
        for marker in self.NARRATIVE_ENDS:
            idx = text_lower.find(marker, start_idx + 50)
            if 0 < idx < end_idx:
                end_idx = idx

        narrative = text[start_idx:end_idx].strip()
        narrative = re.sub(r"\s+", " ", narrative)
        return narrative if len(narrative.split()) >= 15 else clean_text(text[:1500])

    def _extract_ipc_sections(self, text: str) -> list:
        sections = set()
        for match in self.IPC_SECTION_PATTERN.finditer(text):
            raw = match.group(1) or match.group(2) or match.group(3) or ""
            for part in re.split(r"[/,]", raw):
                part = part.strip()
                if re.match(r"^\d{1,3}[A-Z]?$", part):
                    try:
                        num = int(re.sub(r"[A-Z]", "", part))
                        if 1 <= num <= 511:
                            sections.add(part)
                    except ValueError:
                        pass
        return sorted(list(sections))


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — EMBEDDING CORPUS BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingCorpusBuilder:
    """Builds the final embedding corpus and training pairs."""

    def build_all(self):
        log("\n" + "="*60)
        log("MODULE 5: Embedding Corpus Construction")
        log("="*60)

        self._build_ipc_corpus()
        self._build_embedding_pairs()
        self._build_final_training_datasets()

    def _build_ipc_corpus(self):
        """Merge all IPC sources into one clean corpus."""
        log("\n[EC-1] Building IPC Corpus...")
        all_sections = {}

        # Priority order: PDF parsed > HF dataset > Kaggle
        source_files = [
            f"{PROCESSED['ipc']}/ipc_sections_parsed.json",
            f"{PROCESSED['ipc']}/hf_ipc_sections.json",
            f"{PROCESSED['ipc']}/kaggle_ipc_sections.json",
        ]

        for sf in source_files:
            if not Path(sf).exists():
                continue
            data = load_json(sf)
            items = data.values() if isinstance(data, dict) else data
            for item in items:
                sec = str(item.get("section_number", "")).strip()
                if sec and sec not in all_sections:
                    all_sections[sec] = item

        if not all_sections:
            log("  No parsed IPC files found yet. Run --step ipc_pdf first.", "WARN")
            return

        # Build embedding-ready corpus chunks
        corpus = []
        for sec_num, section in all_sections.items():
            text = section.get("full_text", section.get("description", ""))
            title = section.get("title", SECTION_TITLES.get(sec_num, f"Section {sec_num}"))
            if not text:
                continue

            # Full section chunk
            full_chunk = (
                f"IPC Section {sec_num}: {title}\n"
                f"Chapter: {IPC_CHAPTERS.get(SECTION_CHAPTER_MAP.get(sec_num, ''), '')}\n\n"
                f"{text}"
            )
            corpus.append({
                "chunk_id":       f"ipc_{sec_num}_full",
                "section_number": sec_num,
                "title":          title,
                "text":           clean_text(full_chunk),
                "type":           "full_section",
                "token_estimate": len(full_chunk.split())
            })

            # Sub-chunks for long sections
            if len(text.split()) > 200:
                for i, chunk in enumerate(chunk_text(text, 200, 30)):
                    corpus.append({
                        "chunk_id":       f"ipc_{sec_num}_chunk_{i}",
                        "section_number": sec_num,
                        "title":          title,
                        "text":           chunk,
                        "type":           "sub_chunk",
                        "token_estimate": len(chunk.split())
                    })

        save_json(corpus, f"{PROCESSED['corpus']}/ipc_corpus.json")
        log(f"  IPC corpus: {len(corpus)} chunks from {len(all_sections)} sections")

    def _build_embedding_pairs(self):
        """Build query-context pairs for embedding fine-tuning."""
        log("\n[EC-2] Building Embedding Training Pairs...")
        pairs = []

        # From charge identification data (best source)
        charge_file = f"{PROCESSED['train_ipc']}/charge_identification.json"
        if Path(charge_file).exists():
            records = load_json(charge_file)
            for rec in records:
                facts = rec.get("facts", "")
                sections = rec.get("ipc_sections", [])
                if facts and sections:
                    for sec in sections[:2]:
                        title = SECTION_TITLES.get(sec, f"Section {sec}")
                        pairs.append({
                            "query":            facts[:300],
                            "positive_context": (f"IPC Section {sec} ({title}): "
                                                 f"{facts[:200]}"),
                            "section":          sec,
                            "source":           "charge_id"
                        })

        # From IPC corpus with synthetic queries
        corpus_file = f"{PROCESSED['corpus']}/ipc_corpus.json"
        if Path(corpus_file).exists():
            corpus = load_json(corpus_file)
            for item in corpus:
                if item["type"] == "full_section":
                    sec = item["section_number"]
                    title = item["title"]
                    text = item["text"]

                    # Generate query variations
                    queries = [
                        f"What are the essential ingredients of Section {sec} IPC?",
                        f"When does Section {sec} IPC apply?",
                        f"What is {title} under IPC?",
                        f"Section {sec} IPC punishment and requirements",
                    ]
                    for q in queries:
                        pairs.append({
                            "query":            q,
                            "positive_context": text[:400],
                            "section":          sec,
                            "source":           "ipc_corpus_synthetic"
                        })

        if not pairs:
            log("  No pairs generated. Add source data first.", "WARN")
            return

        # Shuffle and split
        import random
        random.seed(42)
        random.shuffle(pairs)
        split = int(len(pairs) * 0.85)

        train = [{**p, "id": i} for i, p in enumerate(pairs[:split])]
        evalu = [{**p, "id": i} for i, p in enumerate(pairs[split:])]

        save_json(train, f"{PROCESSED['synthetic']}/synthetic_train.json")
        save_json(evalu, f"{PROCESSED['synthetic']}/synthetic_eval.json")

    def _build_final_training_datasets(self):
        """Merge all training data into final datasets."""
        log("\n[EC-3] Building Final Training Datasets...")

        # Merge all Q&A sources
        qa_files = glob.glob(f"{PROCESSED['train_qa']}/*.json")
        all_qa = []
        for f in qa_files:
            data = load_json(f)
            if isinstance(data, list):
                all_qa.extend(data)

        if all_qa:
            save_json(all_qa, f"{PROCESSED['train_qa']}/final_qa_dataset.json")

        # Merge all IPC-specific training
        ipc_files = glob.glob(f"{PROCESSED['train_ipc']}/*.json")
        all_ipc = []
        for f in ipc_files:
            data = load_json(f)
            if isinstance(data, list):
                all_ipc.extend(data)

        if all_ipc:
            save_json(all_ipc, f"{PROCESSED['train_ipc']}/final_ipc_dataset.json")

        log(f"\n  Final Training Summary:")
        log(f"    QA dataset:          {len(all_qa):>6} records")
        log(f"    IPC-specific:        {len(all_ipc):>6} records")
        log(f"    Embedding pairs:     {self._count_file(PROCESSED['synthetic'] + '/synthetic_train.json'):>6} records")


    def _count_file(self, path: str) -> int:
        if Path(path).exists():
            return len(load_json(path))
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — QUALITY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class QualityValidator:
    """Validates preprocessed data quality."""

    def validate_all(self):
        log("\n" + "="*60)
        log("MODULE 6: Quality Validation")
        log("="*60)

        issues = []
        issues.extend(self._validate_ipc_sections())
        issues.extend(self._validate_fir_data())
        issues.extend(self._validate_training_data())
        issues.extend(self._validate_embedding_pairs())

        log(f"\n  Validation complete. Issues found: {len(issues)}")
        for issue in issues:
            log(f"  ⚠ {issue}", "WARN")

        return issues

    def _validate_ipc_sections(self) -> list:
        issues = []
        ipc_file = f"{PROCESSED['corpus']}/ipc_corpus.json"
        if not Path(ipc_file).exists():
            issues.append("IPC corpus not built yet")
            return issues

        data = load_json(ipc_file)
        if len(data) < 50:
            issues.append(f"IPC corpus too small: {len(data)} chunks (expected 200+)")

        empty_texts = [d["chunk_id"] for d in data if not d.get("text")]
        if empty_texts:
            issues.append(f"{len(empty_texts)} IPC chunks have empty text")

        return issues

    def _validate_fir_data(self) -> list:
        issues = []
        fir_file = f"{PROCESSED['fir']}/fir_all.json"
        if not Path(fir_file).exists():
            issues.append("FIR dataset not built yet")
            return issues

        data = load_json(fir_file)
        no_narrative = [d.get("fir_number", "?") for d in data
                        if not d.get("narrative") or len(d.get("narrative", "").split()) < 15]
        if no_narrative:
            issues.append(f"{len(no_narrative)} FIRs have insufficient narrative")

        no_sections = [d.get("fir_number", "?") for d in data
                       if not d.get("applied_ipc_sections")]
        if no_sections:
            issues.append(f"{len(no_sections)} FIRs have no IPC sections extracted")

        return issues

    def _validate_training_data(self) -> list:
        issues = []
        qa_file = f"{PROCESSED['train_qa']}/final_qa_dataset.json"
        if Path(qa_file).exists():
            data = load_json(qa_file)
            if len(data) < 100:
                issues.append(f"QA dataset too small: {len(data)} (expected 500+)")
        else:
            issues.append("QA training dataset not built")
        return issues

    def _validate_embedding_pairs(self) -> list:
        issues = []
        train = f"{PROCESSED['synthetic']}/synthetic_train.json"
        if Path(train).exists():
            data = load_json(train)
            if len(data) < 200:
                issues.append(f"Embedding pairs too few: {len(data)} (expected 500+)")
            short = [d for d in data if len(d.get("positive_context", "").split()) < 10]
            if short:
                issues.append(f"{len(short)} embedding pairs have very short contexts")
        else:
            issues.append("Embedding training pairs not built")
        return issues


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — STATS REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_stats_report():
    log("\n" + "="*60)
    log("PREPROCESSING STATS REPORT")
    log("="*60)

    data_map = {
        "IPC Corpus":              f"{PROCESSED['corpus']}/ipc_corpus.json",
        "FIR Train Set":           f"{PROCESSED['fir']}/fir_train.json",
        "FIR Eval Set":            f"{PROCESSED['eval']}/fir_eval.json",
        "QA Training Data":        f"{PROCESSED['train_qa']}/final_qa_dataset.json",
        "IPC-Specific Training":   f"{PROCESSED['train_ipc']}/final_ipc_dataset.json",
        "Embedding Train Pairs":   f"{PROCESSED['synthetic']}/synthetic_train.json",
        "Embedding Eval Pairs":    f"{PROCESSED['synthetic']}/synthetic_eval.json",
        "IL-TUR Processed":        f"{PROCESSED['train_qa']}/il_tur_processed.json",
        "Bail Judgments QA":       f"{PROCESSED['train_qa']}/bail_judgment_qa.json",
        "Charge ID Pairs":         f"{PROCESSED['train_ipc']}/charge_identification.json",
        "IPC-CAM Train Pairs":     f"{PROCESSED['train_ipc']}/ipc_cam_training_pairs.json",
    }

    total_records = 0
    total_size_kb = 0

    for name, path in data_map.items():
        if Path(path).exists():
            n = len(load_json(path)) if Path(path).stat().st_size < 50_000_000 else "large"
            kb = Path(path).stat().st_size // 1024
            status = "✓"
            if isinstance(n, int):
                total_records += n
            total_size_kb += kb
        else:
            n = 0
            kb = 0
            status = "✗"
        log(f"  [{status}] {name:<35} {str(n):>8} records  |  {kb:>6} KB")

    log(f"\n  TOTAL: {total_records:,} records | {total_size_kb:,} KB ({total_size_kb//1024} MB)")
    log(f"  Log:   {LOG_FILE}")
    log("="*60)

    log("\n  NEXT STEPS:")
    log("  1. python src/embedding/finetune_all.py --task embedding")
    log("  2. python src/embedding/finetune_all.py --task generative --dataset ipc")
    log("  3. python test_ipc_cam.py")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_ipc_pdf():
    log("\n" + "="*60)
    log("MODULE 1: IPC PDF Extraction & Parsing")
    log("="*60)

    pdf_files = glob.glob(f"{RAW['ipc_pdf']}/**/*.pdf", recursive=True)
    txt_files = glob.glob(f"{RAW['ipc_pdf']}/**/*.txt", recursive=True)

    if not pdf_files and not txt_files:
        log(f"  No IPC PDF/TXT found in {RAW['ipc_pdf']}/", "WARN")
        log("  Place your IPC PDF here: data/raw/ipc_corpus/", "WARN")
        log("  Download: https://www.indiacode.nic.in/bitstream/123456789/4219/1/THE-INDIAN-PENAL-CODE-1860.pdf", "WARN")
        return

    all_sections = {}

    for pdf in pdf_files:
        log(f"\n  Parsing: {Path(pdf).name}")
        parser = IPCPDFParser(pdf)
        sections = parser.parse_sections()
        sections = parser.enrich_with_metadata(sections)
        all_sections.update(sections)

    for txt in txt_files:
        log(f"\n  Parsing text file: {Path(txt).name}")
        with open(txt, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        parser = IPCPDFParser(txt)
        parser.raw_text = content
        sections = parser.parse_sections()
        sections = parser.enrich_with_metadata(sections)
        all_sections.update(sections)

    if all_sections:
        save_json(all_sections, f"{PROCESSED['ipc']}/ipc_sections_parsed.json")
        save_json(list(all_sections.values()),
                  f"{PROCESSED['ipc']}/ipc_sections_list.json")
        log(f"\n  Total IPC sections parsed: {len(all_sections)}")
    else:
        log("  No sections extracted. Check PDF format.", "WARN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Legal IPC-RAG — Raw Data Preprocessor"
    )
    parser.add_argument(
        "--all",        action="store_true",
        help="Run all preprocessing modules"
    )
    parser.add_argument(
        "--step",       type=str,
        choices=["ipc_pdf", "huggingface", "github", "fir", "build_corpus", "validate"],
        help="Run a specific preprocessing step"
    )
    parser.add_argument(
        "--stats",      action="store_true",
        help="Print stats report only"
    )
    args = parser.parse_args()

    start = datetime.now()

    if args.stats:
        print_stats_report()

    elif args.step == "ipc_pdf":
        run_ipc_pdf()
        print_stats_report()

    elif args.step == "huggingface":
        HuggingFacePreprocessor().process_all()
        print_stats_report()

    elif args.step == "github":
        GitHubDatasetPreprocessor().process_all()
        print_stats_report()

    elif args.step == "fir":
        FIRDataPreprocessor().process_all()
        print_stats_report()

    elif args.step == "build_corpus":
        EmbeddingCorpusBuilder().build_all()
        print_stats_report()

    elif args.step == "validate":
        QualityValidator().validate_all()
        print_stats_report()

    elif args.all:
        run_ipc_pdf()
        HuggingFacePreprocessor().process_all()
        GitHubDatasetPreprocessor().process_all()
        FIRDataPreprocessor().process_all()
        EmbeddingCorpusBuilder().build_all()
        QualityValidator().validate_all()
        print_stats_report()

    else:
        parser.print_help()
        print("\nQuick start — run all:")
        print("  python preprocess.py --all\n")
        print("Run individual steps:")
        print("  python preprocess.py --step ipc_pdf        # Parse IPC PDF")
        print("  python preprocess.py --step huggingface    # Process HF datasets")
        print("  python preprocess.py --step github         # Process GitHub datasets")
        print("  python preprocess.py --step fir            # Process FIR documents")
        print("  python preprocess.py --step build_corpus   # Build embedding corpus")
        print("  python preprocess.py --step validate       # Validate output quality")
        print("  python preprocess.py --stats               # Show stats only")

    elapsed = (datetime.now() - start).seconds
    log(f"\nTotal preprocessing time: {elapsed}s")
