import sys
import os
from pathlib import Path

# Ensure the src directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.ipc_extractor import IPCSectionExtractor
from src.ipc_cam.ipc_cam import IPCContextualAlignmentModule
from src.rationale.legal_rationale_generator import LegalRationaleGenerator
from src.misuse_detection.misuse_engine import MisuseRiskAssessmentEngine
from src.generation.citizen_response_generator import CitizenResponseGenerator

def run_local_audit(fir_text: str, fir_id: str = "FIR-LOCAL-001"):
    print("\n" + "="*50)
    print(f"⚖️  Starting Local FIR Audit for {fir_id}")
    print("="*50)

    # Define the local paths you provided
    llama_path = "meta-llama/Llama-3.2-3B"
    gist_path = "avsolatorio/GIST-large-Embedding-v0"
    nli_path = "cross-encoder/nli-deberta-v3-small"

    # Step 1: Extract Sections
    print("\n[1/5] Extracting IPC/BNS Sections...")
    extractor = IPCSectionExtractor()
    applied_sections = extractor.extract_mentioned_sections(fir_text)
    
    if not applied_sections:
        print("❌ No sections automatically detected in the text.")
        return
    print(f"✅ Found Sections: {', '.join(applied_sections)}")

    # Step 2: Contextual Alignment (IPC-CAM)
    print("\n[2/5] Running Contextual Alignment Module (IPC-CAM)...")
    print("      (Loading local embeddings, NLI, and Llama 3.2...)")
    cam = IPCContextualAlignmentModule(
        use_local=False,
        local_model_path=llama_path,
        embedding_model_name=gist_path,
        nli_model_name=nli_path,
        groq_api_key=os.environ.get("GROQ_API_KEY", "")
    )
    cam_report = cam.generate_full_cam_report(fir_id, applied_sections, fir_text)
    print(f"✅ Alignment Score: {sum(s.alignment_score for s in cam_report.sections_evaluated)/len(cam_report.sections_evaluated)*100:.1f}%")

    # Step 3: Rationale Generation
    print("\n[3/5] Generating Legal Rationales...")
    rationale_gen = LegalRationaleGenerator()
    fir_rationale = rationale_gen.generate_fir_level_rationale(cam_report)
    print(f"✅ Overall Risk Level: {fir_rationale.overall_misuse_risk}")

    # Step 4: Misuse Detection
    print("\n[4/5] Running Misuse Risk Engine...")
    misuse_engine = MisuseRiskAssessmentEngine()
    misuse_report = misuse_engine.generate_misuse_report(fir_id, cam_report, fir_rationale)
    print(f"✅ Detected {len(misuse_report.risk_assessment['misuse_patterns'])} malpractice pattern(s).")

    # Step 5: Citizen Response Generation
    print("\n[5/5] Generating Final Citizen Report via Local Llama 3.2...")
    response_gen = CitizenResponseGenerator(
        use_local=False,
        api_key=os.environ.get("GROQ_API_KEY", "")
    )
    final_response = response_gen.generate_full_analysis_response(
        fir_id, applied_sections, cam_report, misuse_report, fir_rationale, language="English"
    )

    # Output Results
    print("\n" + "="*50)
    print("📄 FINAL REPORT PREVIEW")
    print("="*50)
    print(final_response.summary_markdown)

    # Save to file
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / f"{fir_id}_local_audit.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(final_response.summary_markdown)
    print(f"\n💾 Full report saved to: {out_file}")

if __name__ == "__main__":
    # You can change this sample text to test different scenarios
    sample_fir_text = """
    On the night of October 12th, the accused had a minor verbal argument with the complainant 
    over a parking space outside their apartment. The accused pushed the complainant once and walked away. 
    The complainant suffered a minor scratch on their arm. The police have registered an FIR 
    under Section 302 of the Indian Penal Code.
    """
    
    run_local_audit(sample_fir_text)
