"""
Legal IPC-RAG — FIR Document Preprocessor
==========================================
Handles PDF/text/scanned FIR documents, extracts structured fields,
and prepares FIR data for IPC-CAM analysis.
"""

import re
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class FIRStatus(Enum):
    VALID = "valid"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


@dataclass
class FIRDocument:
    fir_number: str = ""
    police_station: str = ""
    district: str = ""
    state: str = ""
    date_of_report: str = ""
    date_of_incident: str = ""
    time_of_incident: str = ""
    complainant: str = ""
    complainant_address: str = ""
    accused: list[str] = field(default_factory=list)
    place_of_occurrence: str = ""
    narrative: str = ""
    applied_ipc_sections: list[str] = field(default_factory=list)
    officer_name: str = ""
    officer_rank: str = ""
    witnesses: list[str] = field(default_factory=list)
    raw_text: str = ""
    file_path: str = ""
    status: str = FIRStatus.VALID.value
    validation_errors: list[str] = field(default_factory=list)
    processing_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class FIRPreprocessor:
    """
    Preprocesses FIR documents from multiple formats.
    Supports: PDF (digital), PDF (scanned/OCR), plain text, JSON.
    """

    # ── Regex Patterns ────────────────────────────────────────────────────────
    IPC_SECTION_PATTERNS = [
        r"[Uu]/[Ss]\s*(\d+[A-Za-z]?(?:/\d+[A-Za-z]?)*)",        # u/s 302/34
        r"[Ss]ection\s+(\d+[A-Za-z]?(?:\s*(?:and|&|,|/)\s*\d+[A-Za-z]?)*)\s+(?:of\s+)?IPC",
        r"IPC\s+[Ss]ection\s*(\d+[A-Za-z]?)",
        r"(?:under|u/s|Under)\s+[Ss]ec(?:tion)?\.?\s*(\d+[A-Za-z]?)",
        r"(\d+[A-Za-z]?)\s+IPC",
        r"[Ss]ection\s+(\d+[A-Za-z]?)\s+of\s+(?:the\s+)?Indian\s+Penal\s+Code",
    ]

    FIR_NUMBER_PATTERNS = [
        r"FIR\s*No\.?\s*:?\s*([\d/\-]+(?:/\d{4})?)",
        r"First\s+Information\s+Report\s+No\.?\s*:?\s*([\d/\-]+)",
        r"F\.I\.R\.?\s*No\.?\s*:?\s*([\d/\-]+)",
        r"Crime\s+No\.?\s*:?\s*([\d/\-]+)",
    ]

    POLICE_STATION_PATTERNS = [
        r"(?:Police\s+Station|P\.S\.?|PS)\s*:?\s*([A-Za-z\s]+?)(?:\n|,|District)",
        r"(?:Thana|थाना)\s*:?\s*([A-Za-z\s]+?)(?:\n|,)",
    ]

    DATE_PATTERNS = [
        r"(?:Date\s+of\s+(?:Incident|Occurrence|Offence))\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(?:Date\s+of\s+Report|Reported\s+on)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{4})",
    ]

    NARRATIVE_MARKERS_START = [
        r"(?:Brief\s+)?Facts?\s*(?:of\s+the\s+[Cc]ase)?\s*:?\s*\n",
        r"(?:Description|Statement|Complaint|Narration)\s*:?\s*\n",
        r"(?:Particulars\s+of\s+(?:the\s+)?Offence)\s*:?\s*\n",
        r"(?:Information|Report)\s+(?:given|received)\s*:?\s*\n",
    ]

    NARRATIVE_MARKERS_END = [
        r"(?:IPC\s+)?[Ss]ections?\s+applied",
        r"(?:Sections?\s+of\s+law\s+applicable)",
        r"(?:Offence(?:s)?\s+committed\s+under)",
        r"Signature\s+of",
        r"Name\s+of\s+(?:the\s+)?Complainant",
        r"Investigating\s+Officer",
    ]

    def __init__(self, ipc_valid_sections: Optional[set] = None):
        self.ipc_valid_sections = ipc_valid_sections or set()

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, input_path: str) -> FIRDocument:
        """Main entry point. Detects format and processes accordingly."""
        path = Path(input_path)
        logger.info(f"Processing FIR: {path.name}")

        raw_text = self._load_text(path)
        fir = self._extract_fields(raw_text, str(path))
        fir = self._validate(fir)
        return fir

    def process_text(self, text: str, fir_id: str = "MANUAL") -> FIRDocument:
        """Process FIR from raw text string."""
        fir = self._extract_fields(text, fir_id)
        fir = self._validate(fir)
        return fir

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="replace")
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("text", data.get("narrative", str(data)))
        elif suffix == ".pdf":
            return self._load_pdf(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_pdf(self, path: Path) -> str:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            text = "\n".join(page.get_text() for page in doc)
            if len(text.strip()) > 100:
                return text
            # Fallback to OCR for scanned PDFs
            return self._ocr_pdf(path)
        except ImportError:
            logger.warning("PyMuPDF not installed. Trying text extraction fallback.")
            return ""

    def _ocr_pdf(self, path: Path) -> str:
        try:
            import fitz
            import pytesseract
            from PIL import Image
            import io
            doc = fitz.open(str(path))
            texts = []
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                texts.append(pytesseract.image_to_string(img, lang="hin+eng"))
            return "\n".join(texts)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    # ── Field Extraction ───────────────────────────────────────────────────────

    def _extract_fields(self, raw_text: str, file_path: str = "") -> FIRDocument:
        fir = FIRDocument(raw_text=raw_text, file_path=file_path)

        fir.fir_number = self._extract_fir_number(raw_text)
        fir.police_station = self._extract_police_station(raw_text)
        fir.district, fir.state = self._extract_location(raw_text)
        fir.date_of_report, fir.date_of_incident = self._extract_dates(raw_text)
        fir.time_of_incident = self._extract_time(raw_text)
        fir.complainant = self._extract_complainant(raw_text)
        fir.accused = self._extract_accused(raw_text)
        fir.place_of_occurrence = self._extract_place(raw_text)
        fir.applied_ipc_sections = self._extract_ipc_sections(raw_text)
        fir.narrative = self._extract_narrative(raw_text)
        fir.officer_name, fir.officer_rank = self._extract_officer(raw_text)
        fir.witnesses = self._extract_witnesses(raw_text)

        fir.processing_metadata = {
            "char_count": len(raw_text),
            "word_count": len(raw_text.split()),
            "narrative_word_count": len(fir.narrative.split()),
            "sections_found": len(fir.applied_ipc_sections),
            "processed_at": datetime.now().isoformat(),
        }
        return fir

    def _extract_fir_number(self, text: str) -> str:
        for pattern in self.FIR_NUMBER_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return "UNKNOWN"

    def _extract_police_station(self, text: str) -> str:
        for pattern in self.POLICE_STATION_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return ""

    def _extract_location(self, text: str) -> tuple[str, str]:
        district_match = re.search(
            r"District\s*:?\s*([A-Za-z\s]+?)(?:\n|,|State)", text, re.IGNORECASE
        )
        state_match = re.search(
            r"State\s*:?\s*([A-Za-z\s]+?)(?:\n|,|Pin)", text, re.IGNORECASE
        )
        district = district_match.group(1).strip() if district_match else ""
        state = state_match.group(1).strip() if state_match else ""
        return district, state

    def _extract_dates(self, text: str) -> tuple[str, str]:
        dates_found = []
        for pattern in self.DATE_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                dates_found.append(m.group(1))
        report_date = dates_found[0] if dates_found else ""
        incident_date = dates_found[1] if len(dates_found) > 1 else dates_found[0] if dates_found else ""
        return report_date, incident_date

    def _extract_time(self, text: str) -> str:
        m = re.search(
            r"[Tt]ime\s*:?\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm|hrs?)?)", text
        )
        return m.group(1).strip() if m else ""

    def _extract_complainant(self, text: str) -> str:
        patterns = [
            r"(?:Complainant|Informant|Victim)\s*(?:'s\s*)?[Nn]ame\s*:?\s*([A-Z][A-Za-z\s\.]+?)(?:\n|,|S/O|D/O|W/O|Age)",
            r"(?:Name\s+of\s+Complainant)\s*:?\s*([A-Z][A-Za-z\s\.]+?)(?:\n|,)",
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()
        return ""

    def _extract_accused(self, text: str) -> list[str]:
        accused_list = []
        patterns = [
            r"(?:Accused|Suspect|Named\s+Accused)\s*[:\-]?\s*\n?((?:[A-Z][A-Za-z\s\.]+(?:\n|,)){1,5})",
            r"(?:Name\s+of\s+Accused)\s*:?\s*([A-Z][A-Za-z\s\.]+?)(?:\n|,|Age|S/O)",
        ]
        for p in patterns:
            for m in re.finditer(p, text, re.IGNORECASE):
                names = re.split(r"[,\n]", m.group(1))
                accused_list.extend([n.strip() for n in names if len(n.strip()) > 2])
        return list(set(accused_list))[:10]

    def _extract_place(self, text: str) -> str:
        patterns = [
            r"[Pp]lace\s+of\s+(?:[Oo]ccurrence|[Oo]ffence|[Ii]ncident)\s*:?\s*([^\n]+)",
            r"[Oo]ccurred\s+(?:at|near|in)\s+([^\n,\.]+)",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return ""

    def _extract_ipc_sections(self, text: str) -> list[str]:
        """Extract all IPC section numbers from the document."""
        found_sections = set()
        for pattern in self.IPC_SECTION_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                raw = m.group(1)
                # Split compound sections like "302/34" or "302 and 34"
                parts = re.split(r"[/,&\s]+(?:and\s+)?", raw)
                for part in parts:
                    sec = re.sub(r"\s+", "", part.strip())
                    if sec and re.match(r"^\d+[A-Za-z]?$", sec):
                        found_sections.add(sec)
        return sorted(list(found_sections), key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0)

    def _extract_narrative(self, text: str) -> str:
        """Extract the FIR narrative/facts section."""
        start_idx = 0
        for marker in self.NARRATIVE_MARKERS_START:
            m = re.search(marker, text, re.IGNORECASE)
            if m:
                start_idx = m.end()
                break

        end_idx = len(text)
        for marker in self.NARRATIVE_MARKERS_END:
            m = re.search(marker, text[start_idx:], re.IGNORECASE)
            if m:
                end_idx = start_idx + m.start()
                break

        narrative = text[start_idx:end_idx].strip()

        # If no markers found, use the whole text minus first/last 200 chars
        if not narrative or len(narrative) < 50:
            narrative = text[200:-200].strip() if len(text) > 400 else text.strip()

        return self._clean_narrative(narrative)

    def _clean_narrative(self, text: str) -> str:
        """Clean and normalize FIR narrative text."""
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s{3,}", " ", text)
        # Remove page numbers
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        # Normalize quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        # Normalize legal shorthand
        text = re.sub(r"\bu/s\b", "under section", text, flags=re.IGNORECASE)
        text = re.sub(r"\bIPC\b", "Indian Penal Code", text)
        return text.strip()

    def _extract_officer(self, text: str) -> tuple[str, str]:
        patterns = [
            r"(?:Investigating\s+Officer|IO|SHO)\s*:?\s*([A-Z][A-Za-z\s\.]+?)\s*,?\s*(SI|ASI|Inspector|DSP|SP|Constable)",
            r"(SI|ASI|Inspector|DSP|SP)\s+([A-Z][A-Za-z\s\.]+?)(?:\n|,)",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                if p == patterns[0]:
                    return m.group(1).strip(), m.group(2).strip()
                else:
                    return m.group(2).strip(), m.group(1).strip()
        return "", ""

    def _extract_witnesses(self, text: str) -> list[str]:
        witnesses = []
        m = re.search(
            r"[Ww]itness(?:es)?\s*:?\s*\n?((?:[A-Z][A-Za-z\s\.]+(?:\n|,)){1,5})", text
        )
        if m:
            names = re.split(r"[,\n]", m.group(1))
            witnesses = [n.strip() for n in names if len(n.strip()) > 2]
        return witnesses[:5]

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate(self, fir: FIRDocument) -> FIRDocument:
        errors = []
        if len(fir.narrative.split()) < 20:
            errors.append("Narrative too short (< 20 words) — insufficient facts for analysis")
        if not fir.applied_ipc_sections:
            errors.append("No IPC sections found in document")
        if not fir.complainant:
            errors.append("Complainant name not extracted")
        # Validate IPC sections against KB
        if self.ipc_valid_sections:
            invalid = [s for s in fir.applied_ipc_sections if s not in self.ipc_valid_sections]
            if invalid:
                errors.append(f"Invalid IPC sections: {invalid}")

        fir.validation_errors = errors
        if errors:
            fir.status = FIRStatus.INCOMPLETE.value if len(errors) <= 2 else FIRStatus.INVALID.value
        return fir

    def save(self, fir: FIRDocument, output_dir: str = "data/processed/fir_processed"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fname = f"fir_{fir.fir_number.replace('/', '_')}.json"
        path = Path(output_dir) / fname
        path.write_text(fir.to_json(), encoding="utf-8")
        logger.info(f"Saved FIR to {path}")
        return str(path)

    def batch_process(self, input_dir: str, output_dir: str = "data/processed/fir_processed") -> list[FIRDocument]:
        results = []
        for file in Path(input_dir).glob("*"):
            if file.suffix.lower() in [".pdf", ".txt", ".json"]:
                try:
                    fir = self.process(str(file))
                    self.save(fir, output_dir)
                    results.append(fir)
                except Exception as e:
                    logger.error(f"Failed to process {file.name}: {e}")
        logger.info(f"Processed {len(results)} FIRs")
        return results


# ── Demo Usage ────────────────────────────────────────────────────────────────
SAMPLE_FIR = """
FIRST INFORMATION REPORT
FIR No.: 245/2024
Police Station: Kotwali
District: Lucknow
State: Uttar Pradesh

Date of Report: 15-03-2024
Date of Incident: 14-03-2024
Time of Incident: 10:30 PM

Name of Complainant: Ramesh Kumar Sharma
S/O: Late Shyam Lal Sharma
Address: 45, Civil Lines, Lucknow

Name of Accused: Suresh Verma, Mahesh Verma, Ramesh Singh

Place of Occurrence: Near Railway Crossing, Hazratganj, Lucknow

Facts of the Case:
The complainant states that on 14-03-2024 at approximately 10:30 PM, while he was 
returning home from work, three persons namely Suresh Verma, Mahesh Verma and 
Ramesh Singh stopped him near the Railway Crossing at Hazratganj. The accused persons 
surrounded him and demanded his mobile phone and wallet. When the complainant refused, 
the accused Suresh Verma took out a knife and threatened to kill him. All three accused 
then forcibly snatched his mobile phone worth Rs. 15,000 and wallet containing Rs. 8,500 
cash. The accused also hit the complainant on his head with a stone, causing bleeding injury. 
The complainant raised an alarm and the accused persons fled from the scene. The complainant 
was taken to KGMU Hospital where his injuries were examined.

Sections Applied: u/s 392, 323, 506, 34 IPC

Investigating Officer: SI Rajendra Prasad
"""

if __name__ == "__main__":
    preprocessor = FIRPreprocessor()
    fir = preprocessor.process_text(SAMPLE_FIR, "DEMO_FIR")
    print("=== Extracted FIR Fields ===")
    print(f"FIR Number: {fir.fir_number}")
    print(f"Police Station: {fir.police_station}")
    print(f"District: {fir.district}, State: {fir.state}")
    print(f"Date of Incident: {fir.date_of_incident}")
    print(f"Complainant: {fir.complainant}")
    print(f"Accused: {fir.accused}")
    print(f"Applied IPC Sections: {fir.applied_ipc_sections}")
    print(f"Narrative ({len(fir.narrative.split())} words): {fir.narrative[:200]}...")
    print(f"Status: {fir.status}")
    print(f"Validation Errors: {fir.validation_errors}")
