"""
Experiment Runner & Benchmarking Script (Phase 4)
Designs and runs all experiments needed for the conference paper.
Evaluates Embeddings, IPC-CAM Alignment, Misuse Detection, and Reranking.
"""

import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

# Import our novelty modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipc_cam.ipc_cam import IPCContextualAlignmentModule, AlignmentStatus
from misuse_detection.misuse_engine import MisuseRiskAssessmentEngine
from retrieval.ipc_retrieval_pipeline import IPCHybridRetriever

@dataclass
class ExperimentResult:
    experiment_name: str
    metrics: Dict[str, float]
    details: List[Dict[str, Any]]

class ExperimentRunner:
    def __init__(self, 
                 test_data_path: str = "data/evaluation/synthetic_eval.json"):
        self.test_data_path = test_data_path
        self.test_data = self._load_test_data()
        
        # Initialize modules
        self.cam = IPCContextualAlignmentModule()
        self.misuse_engine = MisuseRiskAssessmentEngine()
        self.retriever = IPCHybridRetriever()

    def _load_test_data(self) -> List[Dict[str, Any]]:
        """Load synthetic or human-annotated ground truth data."""
        if not Path(self.test_data_path).exists():
            print(f"Warning: Evaluation data not found at {self.test_data_path}. Creating mock data for testing.")
            return [
                {
                    "fir_text": "The accused hit the victim with a stone during a fight.",
                    "applied_sections": ["323"],
                    "ground_truth_alignment": "FULLY_ALIGNED",
                    "ground_truth_misuse": False
                },
                {
                    "fir_text": "The accused had an argument and slapped the victim once. No injuries.",
                    "applied_sections": ["307"],
                    "ground_truth_alignment": "MISALIGNED",
                    "ground_truth_misuse": True
                }
            ]
        with open(self.test_data_path, "r") as f:
            return json.load(f)

    def run_experiment_1_retrieval(self) -> ExperimentResult:
        """EXPERIMENT 1: IPC Retrieval Benchmarking (Hybrid vs Semantic vs BM25)"""
        print("\n--- Running Experiment 1: Retrieval Benchmarking ---")
        
        hits_hybrid = 0
        hits_semantic = 0
        total = len(self.test_data)
        
        for case in self.test_data:
            query = case["fir_text"]
            target = case["applied_sections"][0]
            
            # Semantic Only
            res_sem = self.retriever.retrieve_ipc_context(query, retrieval_mode='semantic')
            if any(str(r['section_number']) == str(target) for r in res_sem.results[:5]):
                hits_semantic += 1
                
            # Hybrid
            res_hyb = self.retriever.retrieve_ipc_context(query, retrieval_mode='hybrid')
            if any(str(r['section_number']) == str(target) for r in res_hyb.results[:5]):
                hits_hybrid += 1
                
        metrics = {
            "Semantic_Hit@5": hits_semantic / total,
            "Hybrid_Hit@5": hits_hybrid / total,
            "Retrieval_Lift": (hits_hybrid - hits_semantic) / total if total > 0 else 0
        }
        
        return ExperimentResult("Retrieval_Benchmarking", metrics, [])

    def run_experiment_2_alignment(self) -> ExperimentResult:
        """EXPERIMENT 2: IPC-CAM Alignment Accuracy vs Ground Truth"""
        print("\n--- Running Experiment 2: IPC-CAM Alignment Accuracy ---")
        
        correct_verdicts = 0
        total = len(self.test_data)
        details = []
        
        for case in self.test_data:
            narrative = case["fir_text"]
            section = case["applied_sections"][0]
            gt = case["ground_truth_alignment"]
            
            # Run IPC-CAM
            res = self.cam.evaluate_section_alignment(section, narrative)
            
            pred = res.alignment_status.value if hasattr(res.alignment_status, 'value') else str(res.alignment_status)
            
            if pred == gt:
                correct_verdicts += 1
                
            details.append({
                "section": section,
                "predicted": pred,
                "ground_truth": gt,
                "score": res.alignment_score
            })
            
        metrics = {
            "Alignment_Accuracy": correct_verdicts / total if total > 0 else 0,
            "Avg_Alignment_Score": np.mean([d['score'] for d in details])
        }
        
        return ExperimentResult("IPC_CAM_Alignment", metrics, details)

    def run_experiment_3_misuse_detection(self) -> ExperimentResult:
        """EXPERIMENT 3: Misuse Detection Precision/Recall"""
        print("\n--- Running Experiment 3: Misuse Detection Performance ---")
        
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for case in self.test_data:
            narrative = case["fir_text"]
            sections = case["applied_sections"]
            gt_misuse = case["ground_truth_misuse"]
            
            # Run CAM + Misuse Engine
            cam_report = self.cam.generate_full_cam_report("EXP", sections, narrative)
            misuse_score = self.misuse_engine.compute_misuse_risk_score(cam_report)
            
            pred_misuse = misuse_score.risk_level.value in ["HIGH", "MEDIUM"]
            
            if pred_misuse and gt_misuse: tp += 1
            elif pred_misuse and not gt_misuse: fp += 1
            elif not pred_misuse and not gt_misuse: tn += 1
            elif not pred_misuse and gt_misuse: fn += 1
            
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "Misuse_Precision": precision,
            "Misuse_Recall": recall,
            "Misuse_F1": f1,
            "Accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        }
        
        return ExperimentResult("Misuse_Detection", metrics, [])

    def run_experiment_6_latency(self) -> ExperimentResult:
        """EXPERIMENT 6: Component-wise Latency Analysis"""
        print("\n--- Running Experiment 6: Latency Analysis ---")
        
        sample = self.test_data[0]
        
        # 1. Retrieval Latency
        start = time.time()
        self.retriever.retrieve_ipc_context(sample["fir_text"])
        lat_retrieval = time.time() - start
        
        # 2. CAM Latency
        start = time.time()
        self.cam.evaluate_section_alignment(sample["applied_sections"][0], sample["fir_text"])
        lat_cam = time.time() - start
        
        # 3. Total Pipeline Latency
        start = time.time()
        report = self.cam.generate_full_cam_report("LATENCY", sample["applied_sections"], sample["fir_text"])
        from rationale.legal_rationale_generator import LegalRationaleGenerator
        rat_gen = LegalRationaleGenerator()
        rationale = rat_gen.generate_fir_level_rationale(report)
        self.misuse_engine.generate_misuse_report("LATENCY", report, rationale)
        lat_total = time.time() - start
        
        metrics = {
            "Latency_Retrieval_Sec": lat_retrieval,
            "Latency_CAM_Sec": lat_cam,
            "Latency_Total_Backend_Sec": lat_total
        }
        
        return ExperimentResult("Latency_Analysis", metrics, [])

    def save_all_results(self, results: List[ExperimentResult], output_path: str = "data/evaluation/experiment_results.json"):
        """Save results to JSON for paper writing."""
        serialized = [asdict(r) for r in results]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(serialized, f, indent=2)
        print(f"\nFinal experiment results saved to {output_path}")

if __name__ == "__main__":
    runner = ExperimentRunner()
    
    results = []
    results.append(runner.run_experiment_1_retrieval())
    results.append(runner.run_experiment_2_alignment())
    results.append(runner.run_experiment_3_misuse_detection())
    results.append(runner.run_experiment_6_latency())
    
    # Summary Table for console
    print("\n" + "="*50)
    print("      EXPERIMENT SUMMARY TABLE")
    print("="*50)
    for res in results:
        print(f"\nExperiment: {res.experiment_name}")
        for k, v in res.metrics.items():
            print(f"  - {k}: {v:.4f}")
            
    runner.save_all_results(results)
