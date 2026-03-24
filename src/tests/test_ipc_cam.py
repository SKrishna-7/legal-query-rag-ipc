import json
from src.analysis.ipc_cam_module import IPCCAMModule

# Your working API key
API_KEY = "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"

# Sample FIR data for testing
TEST_CASE = {
    "fir_number": "TEST/002/2024",
    "narrative": "The investigation revealed that the three accused persons namely Anil, Sunil and Vijay met at a hotel last week. They planned to rob the local bank and even acquired a getaway vehicle and masks for this purpose. They were arrested while they were discussing the execution of the plan.",
    "applied_ipc_sections": ["120B"] # Criminal Conspiracy
}

def test_cam():
    print("Initializing IPC-CAM Module...")
    cam = IPCCAMModule(api_key=API_KEY)
    
    print(f"Analyzing consistency for FIR {TEST_CASE['fir_number']}...")
    print(f"Target Section: {TEST_CASE['applied_ipc_sections'][0]}")
    
    result = cam.analyze_full_fir(TEST_CASE)
    
    print("\n" + "="*50)
    print("IPC-CAM ANALYSIS RESULTS")
    print("="*50)
    print(f"Overall Consistency Score: {result['overall_consistency_score']}")
    print(f"Potential Misuse Detected: {result['potential_misuse_detected']}")
    
    for sec_res in result['section_results']:
        print(f"\n[Section {sec_res['section']}] Assessment: {sec_res['overall_assessment']}")
        print("\nIngredient Breakdown:")
        for ing in sec_res['ingredient_analysis']:
            print(f"- {ing['ingredient']}: [{ing['status']}]")
            print(f"  Evidence: {ing['evidence']}")
            
    print("\nFull JSON Response:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_cam()
