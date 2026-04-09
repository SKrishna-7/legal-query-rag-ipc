import json
from src.analysis.ipc_cam_module import IPCCAMModule
from src.analysis.misuse_detection import MisuseDetectionEngine
from src.analysis.rationale_generator import LegalRationaleGenerator

# Your working API key
API_KEY = ""

# Sample FIR data for testing (Conspiracy case with missing elements)
TEST_CASE = {
    "fir_number": "TEST/003/2024",
    "narrative": "The investigation revealed that the three accused persons namely Anil, Sunil and Vijay met at a hotel last week. They planned to rob the local bank and even acquired a getaway vehicle and masks for this purpose. They were arrested while they were discussing the execution of the plan.",
    "applied_ipc_sections": ["120B"] # Criminal Conspiracy
}

def test_full_pipeline():
    print("1. Initializing Modules...")
    cam = IPCCAMModule(api_key=API_KEY)
    misuse_engine = MisuseDetectionEngine()
    generator = LegalRationaleGenerator(api_key=API_KEY)
    
    print("\n2. Running IPC-CAM (Extracting satisfied/missing ingredients)...")
    cam_report = cam.analyze_full_fir(TEST_CASE)
    
    print("\n3. Running Misuse Detection Engine (Classifying the risk)...")
    full_report = misuse_engine.process_full_report(cam_report)
    
    print(f"Classification result: {full_report['section_results'][0]['misuse_classification']['category']}")
    
    print("\n4. Generating Final Legal Rationale Report...")
    markdown_report = generator.generate_report(full_report)
    
    print("\n" + "="*80)
    print("FINAL LEGAL AUDIT REPORT (MARKDOWN)")
    print("="*80 + "\n")
    print(markdown_report)

if __name__ == "__main__":
    test_full_pipeline()
