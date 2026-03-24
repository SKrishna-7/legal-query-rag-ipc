import os
import json
import logging
from pathlib import Path
from fir_preprocessor import FIRPreprocessor
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
RAW_FIR_DIR = "data/raw/fir_documents/icdar2023"
RAW_BAIL_DIR = "data/raw/fir_documents/bail_judgments"
PROCESSED_OUTPUT_DIR = "data/processed/fir_processed"

def process_icdar_firs(preprocessor):
    """Process real FIRs from ICDAR2023 repository metadata."""
    logger.info("Starting processing of ICDAR2023 FIRs from metadata...")
    
    json_path = os.path.join(RAW_FIR_DIR, "FIR_details.json")
    if not os.path.exists(json_path):
        logger.warning(f"ICDAR details JSON not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Group snippets by image_id
    grouped_data = {}
    for snippet in data:
        img_id = snippet.get("image_id")
        if img_id not in grouped_data:
            grouped_data[img_id] = {
                "name": snippet.get("image_name", f"ID_{img_id}"),
                "snippets": []
            }
        grouped_data[img_id]["snippets"].append(snippet)

    count = 0
    for img_id, info in tqdm(grouped_data.items(), desc="Reconstructing ICDAR FIRs"):
        try:
            # Sort snippets by bbox (y then x) to reconstruct reading order roughly
            sorted_snippets = sorted(info["snippets"], key=lambda x: (x["bbox"][1], x["bbox"][0]))
            full_text = " ".join([s.get("text", "") for s in sorted_snippets])
            
            if not full_text.strip(): continue

            fir_doc = preprocessor.process_text(full_text, fir_id=f"ICDAR_{img_id}")
            
            # Tag the source and use image name as FIR number if unknown
            fir_doc.processing_metadata["source_dataset"] = "icdar2023"
            if fir_doc.fir_number == "UNKNOWN":
                fir_doc.fir_number = info["name"].replace(".jpg", "").replace(" ", "_")
                
            preprocessor.save(fir_doc, os.path.join(PROCESSED_OUTPUT_DIR, "icdar"))
            count += 1
        except Exception as e:
            logger.error(f"Error processing ICDAR record {img_id}: {e}")
            
    logger.info(f"Successfully processed {count} ICDAR FIRs.")

def process_bail_judgments(preprocessor):
    """Process bail judgments as FIR-like fact narratives."""
    logger.info("Starting processing of Bail Judgments...")
    
    json_path = os.path.join(RAW_BAIL_DIR, "indian_bail_judgments.json")
    if not os.path.exists(json_path):
        logger.warning(f"Bail judgments JSON not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    count = 0
    for i, record in enumerate(tqdm(data, desc="Bail Judgments")):
        try:
            # Bail judgments have 'facts' and 'ipc_sections' fields
            facts = record.get("facts", "")
            if not facts: continue
            
            case_id = record.get('case_id', f"GEN_{i}")
            fir_doc = preprocessor.process_text(facts, fir_id=f"BAIL_{case_id}")
            
            # Ensure unique FIR number for filename
            fir_doc.fir_number = f"BAIL_{case_id}"
            fir_doc.police_station = record.get("court", "Bail Judgment")
            fir_doc.date_of_incident = record.get("date", "")
            fir_doc.processing_metadata["source_dataset"] = "bail_judgments_1200"
            
            preprocessor.save(fir_doc, os.path.join(PROCESSED_OUTPUT_DIR, "bail"))
            count += 1
        except Exception as e:
            logger.error(f"Error processing bail record {i}: {e}")

    logger.info(f"Successfully processed {count} Bail Judgments.")

def main():
    # Initialize preprocessor
    # We could load valid IPC sections here if needed for validation
    preprocessor = FIRPreprocessor()
    
    # Create main output directory
    Path(PROCESSED_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Run tasks
    process_icdar_firs(preprocessor)
    process_bail_judgments(preprocessor)
    
    logger.info("Batch processing complete.")

if __name__ == "__main__":
    main()
