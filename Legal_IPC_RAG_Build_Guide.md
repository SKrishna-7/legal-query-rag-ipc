# Legal IPC-RAG: Complete Build Guide
## From Base Paper Reproduction → Proposed Architecture Implementation

---
## 🚀 PROGRESS SUMMARY (Updated March 2026)

### ✅ What We Have Done So Far
We have deliberately skipped Phase 1 (Base Paper Reproduction) to focus entirely on building your core novelties in the **Proposed Architecture (Phase 2)**. 

1. **Phase 2.1 (IPC Knowledge Base):** 
   - Parsed 474 raw IPC sections from local PDFs.
   - Enriched **250 major sections** using the Groq API (Llama 3.3) to extract structured legal elements like "Essential Ingredients" and "Punishments" into individual JSON files.
2. **Phase 2.2 (FIR Document Preprocessing):** 
   - Built a robust `FIRPreprocessor` capable of extracting structured fields (narrative, accused, sections applied) from raw text, PDFs, and JSON.
   - Reconstructed and processed **544 ICDAR2023 FIR images** and **1,200 Bail Judgments**, resulting in **1,744 structured JSON case files**.
3. **Phase 2.4 (IPC Contextual Alignment Module / IPC-CAM):** 
   - Implemented the `IPCCAMModule` (the core novelty) to audit FIR narratives against the enriched IPC Knowledge Base.
   - Successfully verified the module in testing: it accurately audited a Section 120B case, generated a consistency score, and identified missing legal ingredients.
4. **Phase 2.8 (Vector Store / RAG Initialization):** 
   - Initialized ChromaDB using the `multi-qa-mpnet-base-dot-v1` embedding model.
   - Successfully indexed **1,568 processed FIR and bail documents**, creating a massive, semantically searchable database.

### ⏳ What Is Left To Do
The heavy data engineering and core backend are finished. The remaining steps focus on refining the legal analysis and building the application interface.

1. **Phase 2.5 (IPC Section Extraction):** Implement a module (using the downloaded `InLegalNER` dataset) to automatically detect applied IPC sections in FIRs that lack structured headers.
2. **Phase 2.6 (Misuse Risk Assessment Engine):** Build a logic layer to categorize the *type* of inconsistency found by IPC-CAM (e.g., "Over-charging", "Missing Mens Rea", "Fact-Law Mismatch").
3. **Phase 2.7 (Legal Rationale Generator):** Create an LLM-driven module to translate the IPC-CAM JSON outputs into professional, explainable legal audit reports (Markdown/PDF).
4. **Phase 3 (Unified Application Layer):** Build the final CLI, FastAPI, or Streamlit interface (`app.py`) allowing a user to upload a raw FIR and instantly receive a full RAG-backed analysis.
5. **Phase 4 (Benchmarking & Evaluation):** Run the completed system against the downloaded evaluation benchmarks (`IL-TUR` and test sets) to measure its accuracy compared to the baseline paper.
---

> **Purpose:** This document is a step-by-step implementation prompt for building the Legal IPC-RAG system.  
> **Base Paper:** Legal Query RAG (LQ-RAG) — Rahman S. M. Wahidur et al., IEEE Access 2025  
> **Proposed Extension:** Legal IPC-RAG — IPC Misuse Detection & Citizen Legal Guidance Framework  
> **Stack:** Python 3.10+ | LlamaIndex | HuggingFace | FAISS | FastAPI | Streamlit

---

## TABLE OF CONTENTS

```
PHASE 0 — Environment Setup
PHASE 1 — Base Paper Reproduction (LQ-RAG)
  1.1  Dataset Collection & Preprocessing
  1.2  Embedding LLM Fine-Tuning (GIST-Law-Embed)
  1.3  Generative LLM Fine-Tuning (HFM via LoRA)
  1.4  RAG Pipeline Construction
  1.5  Evaluation Agent & Feedback Loop
  1.6  Baseline Evaluation

PHASE 2 — Proposed Architecture (Legal IPC-RAG)
  2.1  IPC Knowledge Base Construction
  2.2  FIR Document Preprocessing Pipeline
  2.3  IPC Section Extraction Module
  2.4  IPC Contextual Alignment Module (IPC-CAM) ← CORE NOVELTY
  2.5  Hybrid Retrieval Pipeline (BM25 + Dense)
  2.6  Fine-Tuned Generative Module for IPC
  2.7  Legal Rationale Generator (Explainability) ← CORE NOVELTY
  2.8  Misuse Risk Assessment Engine ← CORE NOVELTY
  2.9  Citizen-Oriented Response Generation
  2.10 Evaluation Framework

PHASE 3 — API & Interface Layer
PHASE 4 — Experiment Design & Benchmarking
PHASE 5 — Paper Writing Prompts
```

---

# PHASE 0 — ENVIRONMENT SETUP

## 0.1 System Requirements

```
Prompt to Claude / Cursor:
"Set up a Python 3.10 virtual environment for a legal NLP RAG project.
Install the following packages and verify each installation:
- torch==2.1.0 with CUDA 11.8 support
- transformers==4.38.0
- peft==0.8.0
- trl==0.7.11
- llama-index==0.10.0
- llama-index-vector-stores-faiss
- faiss-gpu (or faiss-cpu if no GPU)
- sentence-transformers==2.5.0
- bitsandbytes==0.42.0
- datasets==2.17.0
- evaluate==0.4.1
- rank-bm25==0.2.2
- fastapi==0.109.0
- uvicorn==0.27.0
- streamlit==1.31.0
- PyMuPDF (fitz) for PDF processing
- spacy==3.7.0 with en_core_web_lg model
- pytesseract for OCR on scanned FIRs
- langchain==0.1.0
- openai==1.12.0
- rouge-score
- nltk
- pandas, numpy, matplotlib, seaborn

Create a requirements.txt and a setup.sh script.
Verify GPU availability with torch.cuda.is_available()
Print GPU name and memory if available."
```

## 0.2 Project Directory Structure

```
Prompt:
"Create the following directory structure for the Legal IPC-RAG project.
Add a README.md in each directory explaining its purpose.

legal_ipc_rag/
├── data/
│   ├── raw/
│   │   ├── ipc_corpus/          # Raw IPC documents
│   │   ├── fir_documents/       # Raw FIR PDFs and text files
│   │   └── legal_books/         # Legal reference books from Library Genesis
│   ├── processed/
│   │   ├── ipc_sections/        # Parsed IPC sections as JSON
│   │   ├── fir_processed/       # Cleaned FIR text files
│   │   └── synthetic/           # Synthetic Q&A pairs
│   └── evaluation/
│       ├── test_firs/           # Annotated test FIR set
│       └── ground_truth/        # Expert-labeled IPC misuse annotations
├── models/
│   ├── embedding/               # Fine-tuned embedding models
│   ├── generative/              # Fine-tuned LLaMA checkpoints
│   └── merged/                  # HFM merged model
├── vector_store/
│   ├── ipc_index/               # FAISS index for IPC knowledge base
│   └── legal_index/             # FAISS index for legal corpus
├── src/
│   ├── preprocessing/
│   ├── embedding/
│   ├── retrieval/
│   ├── generation/
│   ├── ipc_cam/                 # IPC Contextual Alignment Module
│   ├── rationale/               # Legal Rationale Generator
│   ├── misuse_detection/        # Misuse Risk Assessment Engine
│   └── evaluation/
├── api/
│   └── routes/
├── ui/
│   └── streamlit_app/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_finetuning.ipynb
│   ├── 03_generative_finetuning.ipynb
│   ├── 04_rag_pipeline.ipynb
│   └── 05_evaluation.ipynb
├── configs/
│   └── config.yaml
└── tests/"
```

---

# PHASE 1 — BASE PAPER REPRODUCTION (LQ-RAG)

> **Goal:** Reproduce the exact LQ-RAG system from the base paper before adding novelties.  
> This validates your infrastructure and gives you baseline numbers for comparison.

---

## 1.1 Dataset Collection & Preprocessing

### 1.1.1 Legal Corpus Collection

```
Prompt:
"Write a Python script: src/preprocessing/corpus_collector.py

This script should:
1. Load legal PDF documents from data/raw/legal_books/ directory
2. Use PyMuPDF (fitz) to extract text from each PDF
3. Handle scanned PDFs using pytesseract OCR as fallback
4. Clean extracted text:
   - Remove page headers, footers, page numbers
   - Remove excessive whitespace and special characters
   - Preserve section numbering and legal references
5. Split text into chunks of 512 tokens with 50-token overlap
   using tiktoken for accurate token counting
6. Save each chunk as: {doc_id}_{chunk_id}.json with fields:
   - chunk_id, doc_id, doc_name, text, token_count, page_number
7. Log statistics: total docs, total chunks, avg chunk size
8. Save full corpus stats to data/processed/corpus_stats.json

Use tqdm for progress bars. Handle encoding errors gracefully."
```

### 1.1.2 Synthetic Dataset Generation (Base Paper Method)

```
Prompt:
"Write a Python script: src/preprocessing/synthetic_data_generator.py

This script replicates the base paper's synthetic data generation using GPT-3.5-turbo.

The script should:
1. Load text chunks from data/processed/
2. For each chunk, call OpenAI GPT-3.5-turbo API with this exact prompt template:
   
   SYSTEM: You are a legal question generator. Given a passage from a legal document,
   generate 3 specific questions that can be answered ONLY from this passage.
   Questions must be legal domain specific, concrete, and non-ambiguous.
   Return ONLY a JSON array of question strings. No explanations.
   
   USER: Passage: {chunk_text}
   
3. Parse the JSON response and validate questions
4. Create query-context pairs: {query, positive_context, chunk_id, doc_id}
5. Split into train (80%) / eval (20%) sets
6. Save to data/processed/synthetic/:
   - synthetic_train.json
   - synthetic_eval.json
7. Handle API rate limits with exponential backoff
8. Track cost and log total tokens used

Target: Generate at least 5,000 query-context pairs for embedding fine-tuning."
```

### 1.1.3 Legal Q&A Dataset Loading

```
Prompt:
"Write a Python script: src/preprocessing/dataset_loader.py

Load and preprocess the following HuggingFace datasets used in the base paper:

1. ibunescu/qa_legal_dataset_train — Legal Q&A (Legal_QA)
   - Filter to English samples only
   - Format: {question, answer, context}

2. tatsu-lab/alpaca — Instruction dataset (cleaned)
   - Load Alpaca-cleaned version
   - Format: {instruction, input, output}

3. rajpurkar/squad_v2 — Reading comprehension
4. truthful_qa — Truthfulness evaluation
5. LegalBench subsets:
   - law_stack_exchange
   - contract_qa
   - abercrombie
   - canada_tax_court_outcomes
   - legal_reasoning_causality

For each dataset:
- Apply data cleaning (remove nulls, duplicates)
- Standardize to unified format: {input_text, target_text, dataset_name, task_type}
- Apply max_length=2048 truncation
- Save to data/processed/{dataset_name}_processed.json

Print dataset statistics table: name, size, avg_input_len, avg_target_len"
```

---

## 1.2 Embedding LLM Fine-Tuning (GIST-Law-Embed)

```
Prompt:
"Write a complete Python training script: src/embedding/finetune_embedding.py

This script fine-tunes the GIST Large Embedding v0 model exactly as described
in the base paper (LQ-RAG) using Multiple Negatives Ranking Loss (MNRL).

CONFIGURATION:
- Base model: 'avsolatorio/GIST-large-Embedding-v0'
- Loss function: MultipleNegativesRankingLoss from sentence_transformers
- Batch sizes to test: [8, 10]
- Epochs to test: [3, 5, 10, 15]
- Learning rate: 2e-5 with warmup ratio 0.1
- Evaluation steps: every 100 steps

TRAINING DATA:
- Load data/processed/synthetic/synthetic_train.json
- Format as InputExample(texts=[query, positive_context])
- Use NoDuplicatesDataLoader for MNRL

EVALUATION:
- Load data/processed/synthetic/synthetic_eval.json
- Compute Hit Rate @ K=[1,3,5,10] after each epoch
- Compute MRR @ K=[1,3,5,10] after each epoch
- Save evaluation results to models/embedding/eval_results.json

TRAINING LOOP:
- Use sentence_transformers.SentenceTransformer.fit()
- Save best checkpoint based on MRR@5 to models/embedding/gist-law-embed/
- Save training curves as PNG to models/embedding/training_curves.png

FINAL EVALUATION:
- Compare before/after fine-tuning on ALL 6 evaluation datasets (Table 1 of base paper)
- Compute average Hit Rate and MRR for each dataset
- Generate comparison table matching Table 6 and Table 7 of the base paper
- Save final results to models/embedding/final_eval_results.json

Add detailed logging with timestamps for each step."
```

---

## 1.3 Generative LLM Fine-Tuning (HFM via LoRA)

### 1.3.1 LoRA Configuration & Training

```
Prompt:
"Write a Python training script: src/generative/finetune_llama.py

Fine-tune LLaMA-3-8B using LoRA (PEFT) exactly as described in the base paper.

CONFIGURATION:
model_name = 'meta-llama/Meta-Llama-3-8B'
LoRA config:
  - r (rank) = 16
  - lora_alpha = 32
  - target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
  - lora_dropout = 0.05
  - bias = 'none'
  - task_type = 'CAUSAL_LM'

Quantization (BitsAndBytesConfig):
  - load_in_4bit = True
  - bnb_4bit_compute_dtype = torch.float16
  - bnb_4bit_quant_type = 'nf4'
  - bnb_4bit_use_double_quant = True

Training Arguments:
  - per_device_train_batch_size = 4
  - gradient_accumulation_steps = 4
  - num_train_epochs = 3
  - learning_rate = 2e-4
  - fp16 = True
  - logging_steps = 10
  - evaluation_strategy = 'steps'
  - eval_steps = 100
  - save_strategy = 'steps'
  - save_steps = 100
  - load_best_model_at_end = True
  - metric_for_best_model = 'eval_loss'
  - early_stopping_patience = 3
  - weight_decay = 0.01

RUN TWICE:
1. Fine-tune on Legal Q&A dataset → save to models/generative/llama_qa/
2. Fine-tune on Instruction dataset → save to models/generative/llama_instr/

Use SFTTrainer from trl library.
Apply chat template for LLaMA-3 tokenizer.
Log loss curves for both runs."
```

### 1.3.2 Model Merging (HFM Creation)

```
Prompt:
"Write a Python script: src/generative/merge_models.py

Merge the two fine-tuned LoRA models using Linear Merging as described in the base paper.

Steps:
1. Load base model: meta-llama/Meta-Llama-3-8B (4-bit quantized)
2. Load LoRA adapter 1: models/generative/llama_qa/
3. Load LoRA adapter 2: models/generative/llama_instr/
4. Merge both adapters into the base model using PEFT merge_and_unload()
5. Implement Linear Merging:
   theta_merged = 0.5 * theta_qa + 0.5 * theta_instr
   (merge at weight level, not adapter level)
6. Save merged model to: models/merged/hfm/
7. Save tokenizer alongside the model
8. Verify merged model can generate text with a test prompt
9. Log model size before and after merging

Also implement Task Arithmetic merging as comparison (PROPOSED NOVELTY):
   tau_qa = theta_qa - theta_base
   tau_instr = theta_instr - theta_base
   theta_ta = theta_base + lambda1 * tau_qa + lambda2 * tau_instr
   Test lambda values: [0.3, 0.5, 0.7] for each
   Save best Task Arithmetic model to models/merged/hfm_ta/"
```

---

## 1.4 RAG Pipeline Construction

```
Prompt:
"Write a complete RAG pipeline: src/retrieval/rag_pipeline.py

Build the full LQ-RAG pipeline using LlamaIndex as described in the base paper.

COMPONENTS TO BUILD:

1. DocumentIngestion class:
   - Load legal corpus from data/raw/legal_books/
   - Use SimpleDirectoryReader with PDFReader
   - Apply SentenceSplitter: chunk_size=512, chunk_overlap=50
   - Embed all chunks using fine-tuned GIST-Law-Embed
   - Build FAISS index and save to vector_store/legal_index/
   - Save metadata: doc_id, chunk_id, source, page_number

2. HybridRetriever class:
   - Dense retrieval: VectorIndexRetriever using FAISS with top_k=15
   - Sparse retrieval: BM25Retriever using rank-bm25
   - Fusion: QueryFusionRetriever combining both with weights [0.5, 0.5]
   - Reranker: SentenceTransformerRerank using 'BAAI/bge-reranker-large'
   - Return top_k=5 re-ranked chunks

3. ReActAgent setup:
   - Create QueryEngineTool wrapping HybridRetriever
   - Use ReActAgent from llama_index.agent
   - Set max_iterations=10

4. PromptTemplate:
   SYSTEM: You are a legal expert assistant. Answer the legal question based ONLY 
   on the provided context. If the answer is not in the context, say 'I cannot 
   find relevant information in the provided legal documents.'
   
   Context: {context}
   Question: {question}
   
   Provide a detailed, accurate answer with references to specific legal sections.

5. RAGPipeline class with methods:
   - build_index(): Run document ingestion
   - query(question: str) -> dict: Full RAG query returning {answer, sources, retrieval_scores}
   - batch_query(questions: list) -> list: Batch processing

6. Add timing instrumentation to measure latency per component
7. Cache embeddings to avoid recomputation"
```

---

## 1.5 Evaluation Agent & Feedback Loop

```
Prompt:
"Write the evaluation agent and feedback loop: src/evaluation/eval_agent.py

This replicates the recursive feedback mechanism from the base paper.

EvaluationAgent class:
  __init__(model='gpt-4', threshold=0.7):
    - Initialize OpenAI client
    - Set evaluation thresholds for 3 metrics

  evaluate(query: str, response: str, context: list[str]) -> EvalResult:
    Use Chain-of-Thought prompting to evaluate 3 dimensions:
    
    PROMPT:
    'You are a strict legal response evaluator. Evaluate the response on 3 criteria.
    Think step by step for each criterion.
    
    Query: {query}
    Retrieved Context: {context}
    Generated Response: {response}
    
    Evaluate:
    1. ANSWER RELEVANCE (0-1): Does the response directly answer the query?
       Think: [reasoning] Score: [0.0-1.0]
    
    2. CONTEXT RELEVANCE (0-1): Is the response grounded in the provided context?
       Think: [reasoning] Score: [0.0-1.0]
    
    3. GROUNDEDNESS (0-1): Are all claims in the response supported by the context?
       Think: [reasoning] Score: [0.0-1.0]
    
    Return ONLY valid JSON:
    {answer_relevance: float, context_relevance: float, groundedness: float, 
     reasoning: {ar: str, cr: str, g: str}}'
    
    Parse JSON response and return EvalResult dataclass

RecursiveFeedbackLoop class:
  __init__(rag_pipeline, eval_agent, prompt_agent, max_iterations=3):
  
  run(query: str) -> FinalResponse:
    1. Get initial response from RAG pipeline
    2. Evaluate with EvaluationAgent
    3. If all scores >= threshold: return response
    4. Else: call PromptAgent to modify query
    5. Repeat up to max_iterations
    6. Return best response across all iterations
    7. Track: iteration_count, score_progression, final_scores

PromptEngineeringAgent class:
  modify_query(original_query: str, eval_result: EvalResult) -> str:
    Use LLM to simplify/restructure query while preserving intent.
    PROMPT: 'Rewrite this legal query to be more specific and retrievable
    while preserving the original intent. Original: {query}
    Issue: {eval_result.reasoning}'

Log all iterations with timestamps and score progression."
```

---

## 1.6 Baseline Evaluation

```
Prompt:
"Write a comprehensive evaluation script: src/evaluation/baseline_eval.py

Evaluate the base paper reproduction across all metrics.

METRICS TO COMPUTE:
1. Hit Rate @ K (K=1,3,5,10,15) — for embedding evaluation
2. MRR @ K — for embedding evaluation
3. Answer Relevance (AR) — cosine similarity between answer and query embeddings
4. Context Relevance (CR) — cosine similarity between context and query
5. Groundedness (G) — NLI-based fact verification score
6. Accuracy — for classification tasks
7. Exact Match (EM) — for QA tasks
8. BLEU Score — sacrebleu library
9. ROUGE-1, ROUGE-L — rouge-score library

EVALUATION DATASETS:
- BBH (complex reasoning)
- Hellaswag (commonsense)
- TruthfulQA (factuality)
- SQuAD_v2 (reading comprehension)
- MMLU International Law
- MMLU Professional Law
- Abercrombie, LRC, CTCO, CQA (LegalBench)

COMPARISON MODELS:
- Naive RAG (no fine-tuning)
- RAG + FTM (fine-tuned model only, no evaluation agent)
- LQ-RAG (full proposed system)

OUTPUT:
- Generate Table 8 equivalent (reasoning tasks results)
- Generate Table 9 equivalent (legal tasks results)
- Generate Figure 7 equivalent (average relevance bar chart)
- Generate Figure 8 equivalent (RAG triad comparison)
- Save all results to src/evaluation/results/baseline_results.json
- Create evaluation_report.pdf using matplotlib"
```

---

# PHASE 2 — PROPOSED ARCHITECTURE (Legal IPC-RAG)

> **This is your original contribution. Build each module carefully.**  
> The modules below directly address the gaps identified in the base paper  
> and introduce the novelties described in your abstract.

---

## 2.1 IPC Knowledge Base Construction

```
Prompt:
"Write a Python script: src/preprocessing/ipc_knowledge_base.py

Build a structured IPC (Indian Penal Code) knowledge base from raw legal text.

DATA SOURCES:
- Download IPC from: https://www.indiacode.nic.in/handle/123456789/2263
- India Kanoon API for case law references
- Download IPC sections as structured text

IPC PARSER - Parse IPC into structured JSON:
For each IPC section, extract:
{
  section_number: '302',
  title: 'Punishment for murder',
  chapter: 'XVI',
  chapter_title: 'Of Offences Affecting the Human Body',
  full_text: '...complete section text...',
  punishment: 'Death or imprisonment for life + fine',
  essential_ingredients: [
    'Causing death of a person',
    'Act done with intention of causing death',
    'Act done with knowledge that it is likely to cause death'
  ],
  related_sections: ['299', '300', '301', '304'],
  cognizable: True,
  bailable: False,
  triable_by: 'Court of Sessions',
  compoundable: False,
  keywords: ['murder', 'death', 'intention', 'knowledge'],
  case_examples: [],  # Will be populated from case law
  mens_rea_required: True,
  actus_reus: 'Causing death',
  maximum_punishment: 'Death',
  minimum_punishment: 'Life imprisonment'
}

Save complete IPC knowledge base to:
- data/processed/ipc_sections/ipc_complete.json (all sections)
- data/processed/ipc_sections/ipc_section_{N}.json (individual files)

VALIDATION:
- Verify all 511 IPC sections are parsed
- Check essential_ingredients are non-empty for all sections
- Generate completeness report"
```

### 2.1.1 IPC Vector Index Construction

```
Prompt:
"Write a script: src/embedding/build_ipc_index.py

Build the FAISS vector index for IPC knowledge base.

Steps:
1. Load all IPC sections from data/processed/ipc_sections/ipc_complete.json
2. Create rich text representation for each section:
   
   embedding_text = f'''
   IPC Section {section_number}: {title}
   Chapter: {chapter_title}
   
   Legal Text: {full_text}
   
   Essential Ingredients Required to Invoke This Section:
   {chr(10).join([f'{i+1}. {ing}' for i, ing in enumerate(essential_ingredients)])}
   
   Related Sections: {', '.join(related_sections)}
   Keywords: {', '.join(keywords)}
   Punishment: {punishment}
   '''
   
3. Embed all sections using fine-tuned GIST-Law-Embed
4. Build TWO separate FAISS indices:
   a. vector_store/ipc_index/semantic_index — dense semantic search
   b. Also build BM25 index from section texts for lexical search
5. Store metadata: section_number, title, essential_ingredients, punishment
6. Save index to vector_store/ipc_index/
7. Verify retrieval with test queries:
   - 'murder with intention' → should return Section 302
   - 'theft of property' → should return Section 378-382
   - 'assault causing grievous hurt' → should return Section 320-326"
```

---

## 2.2 FIR Document Preprocessing Pipeline

```
Prompt:
"Write a complete FIR preprocessing pipeline: src/preprocessing/fir_preprocessor.py

FIRs are complex legal documents. Build a pipeline that handles them robustly.

FIR STRUCTURE:
A standard Indian FIR contains:
- FIR Number, Date, Police Station, District
- Complainant name, address, contact
- Accused name(s) and description
- Offense details (date, time, place of incident)
- Narrative of facts (most important — this is what we analyze)
- IPC sections applied by the officer
- Witness information
- Officer details

FIRPreprocessor class with methods:

1. load_fir(file_path: str) -> dict:
   - Support PDF (PyMuPDF), text (.txt), and image-based PDFs (pytesseract OCR)
   - Detect if FIR is scanned (image) or digital text
   - For scanned: apply pytesseract with 'hin+eng' language for Hindi/English FIRs

2. extract_fir_fields(raw_text: str) -> FIRDocument:
   Use regex patterns + spaCy NER to extract:
   - fir_number: regex pattern for FIR numbering
   - police_station: named entity + keyword matching
   - date_of_incident: date extraction with dateparser
   - complainant: NER PERSON entities near 'complainant'
   - accused: NER PERSON entities near 'accused'/'defendant'
   - place_of_occurrence: NER GPE/LOC entities
   - narrative: text between 'facts' and 'sections applied' markers
   - applied_ipc_sections: regex '[Ss]ection\s+(\d+[A-Za-z]?)' extraction
   - officer_rank_name: regex for officer information
   
3. clean_narrative(narrative: str) -> str:
   - Normalize Hindi-English code-switched text
   - Standardize legal terminology (e.g., 'complainant' vs 'plaintiff')
   - Remove procedural boilerplate
   - Fix OCR errors using spell correction
   
4. validate_fir(fir_doc: FIRDocument) -> ValidationResult:
   - Check required fields are present
   - Verify IPC sections are valid (exist in IPC knowledge base)
   - Flag if narrative is too short (< 50 words) — insufficient facts
   
5. save_processed_fir(fir_doc: FIRDocument, output_path: str)

DataClass FIRDocument:
  fir_number: str
  police_station: str
  district: str
  state: str
  date_of_report: date
  date_of_incident: date
  complainant: str
  accused: list[str]
  place_of_occurrence: str
  narrative: str  ← MOST IMPORTANT FIELD
  applied_ipc_sections: list[str]
  officer_name: str
  officer_rank: str
  raw_text: str
  processing_metadata: dict"
```

---

## 2.3 IPC Section Extraction Module

```
Prompt:
"Write a module: src/preprocessing/ipc_extractor.py

This module extracts and validates IPC sections from FIR narrative text.
This is the INPUT to the IPC-CAM module.

IPCSectionExtractor class:

1. extract_mentioned_sections(text: str) -> list[str]:
   Extract ALL IPC sections mentioned in text using:
   - Pattern 1: 'Section 302 IPC', 'u/s 302', 'under section 302'
   - Pattern 2: 'IPC 302', '302 IPC'
   - Pattern 3: '302/34 IPC' → extract [302, 34]
   - Pattern 4: Read/A with read-with sections
   - Normalize all to string numbers: ['302', '34', '120B']
   - Cross-reference with IPC knowledge base to validate

2. extract_facts_as_claims(narrative: str) -> list[Claim]:
   Use spaCy to extract factual claims from FIR narrative:
   
   For each sentence in narrative:
   - Extract Subject-Verb-Object triplets
   - Classify claim type: ACTION, STATE, INTENTION, KNOWLEDGE, RELATIONSHIP
   - Extract temporal markers (when)
   - Extract location markers (where)
   - Assess claim certainty: CERTAIN / ALLEGED / UNKNOWN
   
   Return list of Claim objects:
   Claim(
     text: str,
     subject: str,
     predicate: str,
     object: str,
     claim_type: ClaimType,
     temporal: str,
     location: str,
     certainty: CertaintyLevel
   )

3. map_facts_to_section_elements(
     claims: list[Claim], 
     ipc_sections: list[str]
   ) -> FactElementMapping:
   
   For each IPC section:
   - Load essential_ingredients from IPC knowledge base
   - For each ingredient, find matching claims from narrative
   - Compute similarity score using GIST-Law-Embed
   - Return mapping: {section: {ingredient: [matching_claims]}}
   
   This mapping is the PRIMARY INPUT to IPC-CAM.

4. Save extraction results with confidence scores"
```

---

## 2.4 IPC Contextual Alignment Module (IPC-CAM) ← CORE NOVELTY

> **This is your primary novel contribution. Build this carefully.**

```
Prompt:
"Write the core novel module: src/ipc_cam/ipc_cam.py

IPC Contextual Alignment Module (IPC-CAM) is the primary novelty of this paper.
It evaluates whether the facts in a FIR legally satisfy the requirements to invoke
specific IPC sections.

CORE IDEA:
Every IPC section has 'essential ingredients' — legal elements that MUST ALL be 
present in the facts to lawfully invoke that section. IPC-CAM checks each ingredient
against the FIR narrative systematically.

IPCContextualAlignmentModule class:

1. compute_ingredient_satisfaction_score(
     ingredient: str,
     fir_narrative: str,
     retrieved_context: list[str]
   ) -> IngredientScore:
   
   Use a 3-step process:
   
   Step A — Semantic Similarity:
   - Embed ingredient text using GIST-Law-Embed
   - Embed each sentence of FIR narrative
   - Compute cosine similarity scores
   - Top-3 matching sentences become 'evidence sentences'
   
   Step B — NLI Entailment Check:
   - Use 'cross-encoder/nli-deberta-v3-large' NLI model
   - Check: does evidence_sentences ENTAIL ingredient requirement?
   - Get: {entailment_score, neutral_score, contradiction_score}
   
   Step C — LLM Verification:
   - Prompt HFM with:
     'Legal Ingredient: {ingredient}
      FIR Facts: {evidence_sentences}
      Question: Do the FIR facts establish the legal requirement stated above?
      Answer with: SATISFIED / PARTIALLY_SATISFIED / NOT_SATISFIED
      Confidence: [0.0-1.0]
      Reasoning: [brief legal reasoning]'
   
   Combine A+B+C into final IngredientScore:
   IngredientScore(
     ingredient: str,
     satisfaction_status: SatisfactionStatus,  # SATISFIED/PARTIAL/NOT_SATISFIED
     confidence: float,
     semantic_score: float,
     nli_entailment: float,
     llm_verdict: str,
     evidence_sentences: list[str],
     reasoning: str
   )

2. evaluate_section_alignment(
     section_number: str,
     fir_doc: FIRDocument,
     retrieved_ipc_context: list[str]
   ) -> SectionAlignmentResult:
   
   - Load all essential_ingredients for this IPC section
   - For each ingredient: compute_ingredient_satisfaction_score()
   - Compute overall alignment:
     alignment_score = weighted_avg(ingredient_scores)
     
   - Classify alignment:
     FULLY_ALIGNED: all ingredients satisfied (score >= 0.80)
     PARTIALLY_ALIGNED: some ingredients satisfied (score 0.50-0.79)
     MISALIGNED: critical ingredients missing (score < 0.50)
   
   - Identify MISSING ingredients (not satisfied)
   - Identify PARTIAL ingredients (partially satisfied)
   
   Return SectionAlignmentResult(
     section_number: str,
     alignment_status: AlignmentStatus,
     alignment_score: float,
     ingredient_scores: list[IngredientScore],
     missing_ingredients: list[str],
     partial_ingredients: list[str],
     alignment_reasoning: str
   )

3. evaluate_all_applied_sections(
     fir_doc: FIRDocument,
     retrieved_context: list[str]
   ) -> IPCCAMReport:
   
   - Run evaluate_section_alignment() for EVERY section in fir_doc.applied_ipc_sections
   - Additionally: suggest potentially MISSING sections that should have been applied
     (based on FIR facts that match unapplied sections)
   - Identify OVER-SEVERE sections (applied section too harsh for stated facts)
   - Identify UNDER-APPLIED sections (lighter section should have been used)
   
   Return IPCCAMReport(
     fir_number: str,
     sections_evaluated: list[SectionAlignmentResult],
     misuse_detected: bool,
     misuse_type: list[MisuseType],  # OVER_SEVERITY, WRONG_SECTION, MISSING_ELEMENT
     suggested_sections: list[str],
     overall_misuse_risk_score: float,
     cam_summary: str
   )

4. detect_misuse_patterns(cam_report: IPCCAMReport) -> MisuseAnalysis:
   
   MISUSE PATTERNS TO DETECT:
   a. Over-severity: Section applied is more serious than facts support
      (e.g., Section 302 murder applied when facts only show 304 culpable homicide)
   b. Missing mens rea: Section requires criminal intent but FIR shows no intent evidence
   c. Missing actus reus: Section requires specific act but FIR narrative doesn't describe it
   d. Cognizable vs Non-cognizable mismatch: Wrong procedure used
   e. Jurisdiction error: Section doesn't apply in the jurisdiction
   f. Related section omission: Companion sections (e.g., 34 - common intention) wrongly applied
   
   For each detected pattern:
   - Pattern type
   - Severity: HIGH / MEDIUM / LOW
   - Affected sections
   - Legal basis for detection
   - Recommended correction"
```

---

## 2.5 Hybrid Retrieval Pipeline (IPC-Specific)

```
Prompt:
"Write the IPC-specific retrieval pipeline: src/retrieval/ipc_retrieval_pipeline.py

Extend the base RAG retrieval for IPC-specific legal retrieval.

IPCHybridRetriever class:

1. retrieve_ipc_context(
     query: str,
     section_numbers: list[str] = None,
     retrieval_mode: str = 'hybrid'
   ) -> RetrievalResult:
   
   THREE RETRIEVAL MODES:
   
   Mode A — Section-Specific Retrieval:
   If section_numbers provided:
   - Directly load those IPC sections from knowledge base
   - Retrieve case law and precedents for those specific sections
   - Return with source tagging: {source: 'IPC_SECTION', section: '302'}
   
   Mode B — Semantic Retrieval:
   - Embed query using GIST-Law-Embed
   - Search FAISS IPC index with top_k=10
   - Return semantically similar sections
   
   Mode C — Hybrid (DEFAULT):
   - Run Mode A + Mode B simultaneously
   - BM25 search on IPC text corpus
   - Fuse results using Reciprocal Rank Fusion (RRF)
   - Re-rank using cross-encoder
   - Return top_k=5 with diversity filtering

2. retrieve_citizen_explanation_context(section_number: str) -> CitizenContext:
   Special retrieval for citizen-friendly explanations:
   - Retrieve: section text, plain language summary, common examples
   - Retrieve: rights of accused under this section
   - Retrieve: bail conditions and procedures
   - Retrieve: typical punishment range with case examples
   
3. add_temporal_context(section_number: str) -> AmendmentHistory:
   - Retrieve amendment history for IPC section
   - Flag if section has been recently amended (post-2020)
   - Return latest applicable version"
```

---

## 2.6 Fine-Tuned Generative Module for IPC

```
Prompt:
"Write training script: src/generative/finetune_ipc_llm.py

Fine-tune HFM specifically for IPC-related generation tasks.

ADDITIONAL IPC-SPECIFIC FINE-TUNING DATA:

Create a specialized IPC Q&A dataset (data/processed/ipc_qa_dataset.json):

Generate training examples covering these task types:
1. Section Explanation Task:
   Input: 'Explain IPC Section {N} in simple terms for a citizen'
   Output: Plain language explanation + rights + procedure

2. Element Check Task:
   Input: 'Facts: {fir_narrative}. Does this satisfy Section {N}?'
   Output: Structured analysis of each essential ingredient

3. Misuse Detection Task:
   Input: 'FIR applies Section {N}. Facts: {narrative}. Is this appropriate?'
   Output: Misuse analysis with legal reasoning

4. Section Suggestion Task:
   Input: 'Facts: {narrative}. Which IPC sections should apply?'
   Output: Recommended sections with legal justification

5. Citizen Guidance Task:
   Input: 'I received FIR with Section {N}. What are my rights?'
   Output: Rights, bail info, legal procedure, recommended actions

GENERATE 10,000+ examples using GPT-4 with legal expert prompts.

Fine-tune HFM on IPC dataset using same LoRA config as Phase 1.3.
Save to: models/generative/llama_ipc/

Create IPC-HFM by merging:
- llama_ipc adapter (legal IPC knowledge)
- llama_instr adapter (instruction following)
Save final IPC-HFM to: models/merged/ipc_hfm/"
```

---

## 2.7 Legal Rationale Generator (Explainability) ← CORE NOVELTY

```
Prompt:
"Write explainability module: src/rationale/legal_rationale_generator.py

The Legal Rationale Generator makes every decision of the system explainable.
This is the explainable AI component of our novelty.

LegalRationaleGenerator class:

1. generate_section_rationale(
     section_number: str,
     alignment_result: SectionAlignmentResult,
     fir_doc: FIRDocument
   ) -> SectionRationale:
   
   Generate structured legal rationale showing:
   
   RATIONALE FORMAT (JSON):
   {
     section: '302',
     title: 'Punishment for Murder',
     alignment_verdict: 'MISALIGNED',
     alignment_score: 0.42,
     ingredient_analysis: [
       {
         ingredient: 'Causing death of a person',
         status: 'SATISFIED',
         evidence: ['The accused struck the victim with a weapon', 
                   'Victim died on the spot'],
         confidence: 0.91,
         legal_note: 'Actus reus element satisfied'
       },
       {
         ingredient: 'Intention to cause death',
         status: 'NOT_SATISFIED',
         evidence: [],
         confidence: 0.12,
         legal_note: 'No evidence of premeditation or intention stated in FIR narrative'
       }
     ],
     missing_elements: ['Intention to cause death'],
     verdict_reasoning: 'Section 302 requires both actus reus (death caused) AND 
                        mens rea (intention/knowledge). While death occurred, 
                        the FIR narrative contains no facts establishing criminal 
                        intention. Consider Section 304 (culpable homicide) instead.',
     recommended_sections: ['304', '304A'],
     citizen_explanation: 'The police charged you under Section 302 (Murder), 
                          but based on the facts in your FIR, this may be too 
                          severe. Murder requires proof of intention to kill. 
                          Section 304 (Culpable Homicide) may be more appropriate.',
     legal_disclaimer: 'This analysis is for informational purposes only and 
                       does not constitute legal advice. Consult a qualified 
                       advocate for professional legal assistance.'
   }

2. generate_fir_level_rationale(
     cam_report: IPCCAMReport,
     fir_doc: FIRDocument
   ) -> FIRRationale:
   
   Aggregate section-level rationales into FIR-level summary:
   
   - Overall misuse risk assessment (HIGH/MEDIUM/LOW with score)
   - Per-section verdict summary table
   - List of concerning sections with specific reasoning
   - List of correctly applied sections
   - Missing sections that should have been added
   - Priority actions for citizen (appeal, bail application, etc.)

3. generate_element_satisfaction_visualization(
     section_rationale: SectionRationale
   ) -> dict:
   
   Generate data for radar/spider chart visualization:
   - Each ingredient as an axis
   - Satisfaction score (0-1) as value
   - Return chart data as JSON for frontend rendering

4. generate_plain_language_summary(
     fir_rationale: FIRRationale,
     target_audience: str = 'citizen'  # or 'lawyer', 'researcher'
   ) -> str:
   
   Use IPC-HFM to generate audience-appropriate summary:
   - citizen: Simple language, avoid jargon, focus on rights and next steps
   - lawyer: Technical analysis, section-specific reasoning, procedure
   - researcher: Statistical summary, confidence scores, methodology notes"
```

---

## 2.8 Misuse Risk Assessment Engine ← CORE NOVELTY

```
Prompt:
"Write the misuse detection engine: src/misuse_detection/misuse_engine.py

This module computes the final misuse risk score and generates actionable alerts.

MisuseRiskAssessmentEngine class:

MISUSE TAXONOMY (define as Enum):
class MisuseType(Enum):
    OVER_SEVERITY = 'More serious section than facts support'
    UNDER_SEVERITY = 'Less serious section, likely to compromise prosecution'
    MISSING_MENS_REA = 'Section requires criminal intent not shown in facts'
    MISSING_ACTUS_REUS = 'Section requires specific act not described in FIR'
    WRONG_JURISDICTION = 'Section not applicable in this jurisdiction'
    PROCEDURAL_MISMATCH = 'Cognizable/non-cognizable classification error'
    COMPANION_SECTION_MISUSE = 'Joint liability section (34/120B) wrongly applied'
    BAIL_MANIPULATION = 'Non-bailable section applied to deny bail inappropriately'

1. compute_misuse_risk_score(
     cam_report: IPCCAMReport,
     fir_doc: FIRDocument
   ) -> MisuseRiskScore:
   
   SCORING ALGORITHM:
   
   base_score = 0.0
   weights = {
     OVER_SEVERITY: 0.30,
     MISSING_MENS_REA: 0.25,
     MISSING_ACTUS_REUS: 0.20,
     PROCEDURAL_MISMATCH: 0.15,
     COMPANION_SECTION_MISUSE: 0.10
   }
   
   For each misuse pattern detected:
     base_score += weights[misuse_type] * pattern_severity_multiplier
   
   alignment_penalty = 1 - avg(section_alignment_scores)
   
   final_score = min(1.0, base_score + 0.3 * alignment_penalty)
   
   risk_level = 
     HIGH: final_score >= 0.70
     MEDIUM: final_score >= 0.40
     LOW: final_score < 0.40
   
   Return MisuseRiskScore(
     overall_score: float,
     risk_level: RiskLevel,
     misuse_patterns: list[DetectedPattern],
     section_risk_scores: dict[str, float],
     confidence_interval: tuple[float, float],
     risk_reasoning: str
   )

2. generate_citizen_alert(
     misuse_risk: MisuseRiskScore,
     fir_doc: FIRDocument
   ) -> CitizenAlert:
   
   HIGH RISK alert triggers:
   - Specific sections to challenge with reasoning
   - Recommended immediate actions (bail application, anticipatory bail)
   - Rights under CrPC (Code of Criminal Procedure)
   - Suggested questions to ask the advocate
   
   MEDIUM RISK alert triggers:
   - Sections worth reviewing with advocate
   - Procedural checks to perform
   - Documentation to gather
   
   LOW RISK alert:
   - General legal awareness information
   - Confirmation of properly applied sections

3. generate_misuse_report(
     fir_doc: FIRDocument,
     cam_report: IPCCAMReport,
     misuse_risk: MisuseRiskScore,
     fir_rationale: FIRRationale
   ) -> MisuseReport:
   
   Complete structured report containing:
   - Executive summary (1 paragraph)
   - Risk dashboard (scores per section)
   - Detailed section analysis table
   - Misuse patterns found
   - Legal recommendations
   - Citizen action guide
   - Legal disclaimer

4. Save report to: data/evaluation/misuse_reports/{fir_number}_report.json"
```

---

## 2.9 Citizen-Oriented Response Generation

```
Prompt:
"Write the response generation module: src/generation/citizen_response_generator.py

This module produces the final citizen-facing output — simplified, actionable, and clear.

CitizenResponseGenerator class:

1. generate_full_analysis_response(
     fir_doc: FIRDocument,
     cam_report: IPCCAMReport,
     misuse_report: MisuseReport,
     fir_rationale: FIRRationale,
     target_audience: str = 'citizen'
   ) -> CitizenResponse:
   
   Use IPC-HFM with the following structured prompt:
   
   SYSTEM PROMPT:
   'You are a legal awareness assistant helping Indian citizens understand 
   their FIR documents. You NEVER give legal advice or act as a lawyer. 
   You explain legal concepts in simple terms and help citizens understand 
   their rights. Always recommend consulting a qualified advocate for legal decisions.
   
   Respond in {language}. Use simple, clear language. Avoid complex legal jargon.
   Structure your response with clear headings.'
   
   USER PROMPT:
   'FIR Analysis Request:
   FIR Number: {fir_number}
   Applied IPC Sections: {applied_sections}
   
   TECHNICAL ANALYSIS (for your reference only, do not show to user):
   {cam_report_summary}
   {misuse_risk_summary}
   
   Generate a citizen-friendly analysis covering:
   1. What each IPC section means in simple terms
   2. What elements the police need to prove for each section
   3. Whether the facts described in the FIR support each section [YES/PARTIAL/UNCLEAR]
   4. Any concerning patterns found in section application
   5. Your rights under these sections
   6. Immediate steps you can take
   7. Questions to ask your lawyer
   
   IMPORTANT: End with the disclaimer that this is for awareness only.'
   
   Response structure:
   CitizenResponse(
     fir_number: str,
     section_summaries: list[dict],  # Plain language per section
     overall_assessment: str,  # 2-3 sentence summary
     risk_indicator: str,  # 'Review Recommended' / 'Concerning' / 'Appears Standard'
     citizen_rights: list[str],
     immediate_actions: list[str],
     questions_for_lawyer: list[str],
     disclaimer: str,
     language: str,
     generation_timestamp: datetime
   )

2. Support MULTILINGUAL output:
   Supported languages: ['en', 'hi', 'mr', 'ta', 'te', 'bn', 'gu', 'kn']
   Use IPC-HFM for English + translation API for other languages
   
3. generate_summary_card(citizen_response: CitizenResponse) -> dict:
   Short summary for UI display (max 150 words + risk badge)"
```

---

## 2.10 Evaluation Framework (Novel Contribution Evaluation)

```
Prompt:
"Write comprehensive evaluation framework: src/evaluation/ipc_evaluation.py

Evaluate the Legal IPC-RAG system's novel contributions.

EVALUATION DATASETS NEEDED:
1. Annotated FIR Test Set (data/evaluation/test_firs/):
   - Minimum 100 FIR documents
   - Annotated by at least 2 legal practitioners
   - Labels: {fir_id, applied_sections, correct_sections, misuse_present, misuse_type}
   - Source: Public court records, India Kanoon, anonymized samples

2. IPC Section Alignment Test (data/evaluation/ground_truth/alignment_test.json):
   - 50 pairs of (narrative snippet, IPC section) with human alignment labels
   - Labels: ALIGNED / MISALIGNED with reasoning

3. Citizen Comprehension Test:
   - 30 generated responses rated by laypeople (1-5 comprehensibility)

METRICS TO COMPUTE:

For IPC-CAM (Alignment Module):
- IPC Alignment Accuracy: % of sections correctly classified (aligned/misaligned)
- Ingredient Detection Precision/Recall/F1
- Misuse Detection Precision: TP / (TP + FP)
- Misuse Detection Recall: TP / (TP + FN)
- Misuse Detection F1-Score
- Cohen's Kappa: agreement between IPC-CAM and human legal experts

For Retrieval:
- Hit Rate @ K for IPC sections
- MRR for IPC section retrieval
- Section Recall: % of relevant IPC sections retrieved

For Generation:
- Answer Relevance (AR)
- Context Relevance (CR)  
- Groundedness (G)
- BLEU / ROUGE for section explanations
- Citizen Comprehensibility Score (human evaluation 1-5)

For Misuse Detection:
- False Positive Rate (wrongly flagging valid sections as misuse)
- False Negative Rate (missing actual misuse)
- Risk Score Calibration (does HIGH risk actually mean higher misuse?)

ABLATION STUDY:
Compare variants:
1. Legal IPC-RAG (full system)
2. Without IPC-CAM (only RAG generation, no alignment check)
3. Without Legal Rationale Generator (no explainability)
4. Without IPC fine-tuning (using base LLaMA only)
5. Naive RAG baseline
6. GPT-4 zero-shot baseline

GENERATE:
- Tables matching base paper format (Tables 8, 9, 10, 11 equivalent)
- Bar charts for metric comparisons
- Confusion matrix for misuse detection
- Calibration curve for risk scores
- Inter-rater reliability table (IPC-CAM vs human expert)"
```

---

# PHASE 3 — API & INTERFACE LAYER

## 3.1 FastAPI Backend

```
Prompt:
"Write a production-ready FastAPI application: api/main.py

Build REST API endpoints for Legal IPC-RAG system.

ENDPOINTS:

POST /api/fir/analyze
  Input: multipart/form-data with FIR file (PDF/TXT) or JSON with FIR text
  Process: Full Legal IPC-RAG pipeline
  Output: Complete analysis response with misuse report
  Async: Use background tasks for long processing

POST /api/fir/quick-check
  Input: JSON {fir_text: str, ipc_sections: list[str]}
  Process: Quick IPC-CAM check only (no full generation)
  Output: Alignment scores per section (< 5 seconds)

GET /api/ipc/section/{section_number}
  Output: Full IPC section details with citizen explanation

POST /api/ipc/explain
  Input: {section_number: str, language: str}
  Output: Citizen-friendly explanation in requested language

GET /api/health
  Output: System health, model status, index status

MIDDLEWARE:
- Rate limiting: 10 requests/minute per IP
- Request logging with timing
- Error handling with legal disclaimer on all errors
- Input validation: max FIR size 5MB, max text length 10000 chars

Add OpenAPI documentation with example requests/responses.
Add CORS for frontend access.
Use Pydantic models for all request/response validation."
```

## 3.2 Streamlit Interface

```
Prompt:
"Build a Streamlit UI: ui/streamlit_app/app.py

Create a clean, citizen-friendly interface for Legal IPC-RAG.

PAGES:

1. Home Page:
   - Brief explanation of what the tool does
   - Legal disclaimer (prominent, cannot be missed)
   - Upload FIR document (PDF/image/text)
   - Language selector (English, Hindi + regional languages)

2. FIR Analysis Page:
   - Show extracted FIR fields in a clean table
   - Show identified IPC sections as colored badges
   - Loading spinner with progress steps shown
   - Risk indicator card: 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW

3. Section Analysis Page:
   - Tab per IPC section analyzed
   - For each section:
     * Section title and plain English meaning
     * Ingredient satisfaction radar chart (using plotly)
     * Green ✅ / Yellow ⚠️ / Red ❌ per ingredient
     * Evidence sentences highlighted
     * Alignment verdict badge

4. Misuse Report Page:
   - Full misuse analysis in expandable sections
   - Downloadable PDF report button
   - 'Questions to ask your lawyer' checklist
   - Local legal aid resources (by state)

5. IPC Explorer (Educational):
   - Search any IPC section
   - Show essential ingredients visually
   - Show related sections
   - Show example cases in plain language

STYLING:
- Use Indian government blue/white/orange color scheme
- Mobile-responsive layout
- High contrast text for accessibility
- Always show legal disclaimer at bottom"
```

---

# PHASE 4 — EXPERIMENT DESIGN & BENCHMARKING

```
Prompt:
"Write experiment runner script: src/evaluation/run_experiments.py

Design and run all experiments needed for the conference paper.

EXPERIMENT 1 — Embedding Evaluation (Table 1 equivalent):
Compare on IPC retrieval:
- ColBERT v2
- BGE Embedding (Large)
- GISTEmbed (Large)
- GIST-Law-Embed (base paper fine-tuned)
- IPC-Law-Embed (our further fine-tuned version)
Metrics: Hit Rate @5, @10 | MRR @5, @10

EXPERIMENT 2 — IPC-CAM Alignment Accuracy (Novel):
Compare alignment methods:
- Semantic similarity only (SBERT cosine)
- NLI only (DeBERTa)
- LLM only (GPT-4 zero-shot)
- LLM only (IPC-HFM)
- IPC-CAM (our combined method)
Metric: Alignment Accuracy, F1 vs human annotations

EXPERIMENT 3 — Misuse Detection (Novel):
Compare misuse detection:
- Keyword matching baseline
- GPT-4 zero-shot
- Naive RAG + detection
- Legal IPC-RAG (ours)
Metrics: Precision, Recall, F1, False Positive Rate

EXPERIMENT 4 — End-to-End RAG Quality:
Compare complete systems:
- Naive RAG
- LQ-RAG (base paper system)
- Legal IPC-RAG without IPC-CAM
- Legal IPC-RAG full system
Metrics: AR, CR, Groundedness, BLEU, ROUGE

EXPERIMENT 5 — Citizen Comprehensibility (Novel):
Human evaluation with 10 non-lawyer participants:
- Rate response clarity (1-5)
- Rate response usefulness (1-5)
- Rate trustworthiness (1-5)
Compare: Plain RAG output vs Legal IPC-RAG citizen response

EXPERIMENT 6 — Latency Analysis:
Measure per-component and end-to-end latency:
- FIR preprocessing
- IPC-CAM alignment
- Retrieval (hybrid)
- Generation
- Rationale generation
- Total pipeline

Run all experiments and generate LaTeX-formatted results tables."
```

---

# PHASE 5 — PAPER WRITING PROMPTS

## 5.1 Abstract Writing

```
Prompt to Claude for writing:
"Write a conference-quality abstract for Legal IPC-RAG paper.

Use these exact experimental results: [FILL IN YOUR NUMBERS]
- IPC-CAM alignment accuracy: X%
- Misuse detection F1: X%
- Improvement over Naive RAG: X%
- Improvement over LQ-RAG base: X%
- Citizen comprehensibility score: X/5

The abstract MUST:
- Be 250-300 words
- Contain at least 3 quantitative results
- Clearly state 3 novel contributions (IPC-CAM, Legal Rationale Generator, Risk Engine)
- Not use the word 'project'
- End with demonstrated results, not 'potential'
- Follow ACL 2026 abstract formatting guidelines"
```

## 5.2 Section-by-Section Writing Prompts

```
INTRODUCTION (prompt):
"Write the Introduction section for Legal IPC-RAG paper.
Structure: Problem statement (with IPC misuse statistics from India) →
Limitations of existing work → Our contributions (3 bullet points) →
Paper organization. 
Cite: IPC misuse cases, legal literacy stats in India, existing legal AI papers.
Length: 600-800 words."

RELATED WORK (prompt):
"Write Related Work section covering 4 subsections:
1. Legal AI and LLMs (HanFei, LawGPT, Lawyer LLaMA, LQ-RAG)
2. RAG in Legal Domain (DISC-LawLLM, CBR-RAG, LexDrafter)
3. Indian Legal NLP (gap — limited existing work, our paper fills this)
4. Explainable Legal AI (XAI in legal domain papers)
Clearly state how our work differs from ALL cited papers."

METHODOLOGY (prompt):
"Write detailed Methodology section for Legal IPC-RAG.
Include: System overview diagram description, mathematical formulations for:
- Ingredient Satisfaction Score (combining semantic + NLI + LLM)
- Misuse Risk Score formula
- IPC-CAM alignment computation
- Hybrid retrieval fusion (RRF formula)
Use equation numbering matching base paper style."

EXPERIMENTS (prompt):
"Write Experiments section with all 6 experiments described above.
Include: Dataset statistics table, model configurations table,
all results tables with best numbers in bold,
ablation study table showing contribution of each component."
```

---

# QUICK REFERENCE: BUILD ORDER

```
Week 1-2:  Phase 0 (Setup) + Phase 1.1 (Data)
Week 3-4:  Phase 1.2 (Embedding fine-tuning) + 1.3 (Generative fine-tuning)
Week 5:    Phase 1.4-1.6 (RAG pipeline + Baseline evaluation)
Week 6-7:  Phase 2.1-2.3 (IPC KB + FIR Preprocessing + Section Extraction)
Week 8-9:  Phase 2.4 (IPC-CAM — Core Novelty) ← MOST CRITICAL
Week 10:   Phase 2.5-2.6 (Retrieval + IPC fine-tuning)
Week 11:   Phase 2.7-2.8 (Rationale Generator + Misuse Engine)
Week 12:   Phase 2.9-2.10 (Response Generation + Evaluation)
Week 13:   Phase 3 (API + UI)
Week 14:   Phase 4 (Run ALL experiments + collect numbers)
Week 15:   Phase 5 (Paper writing with real results)
```

---

> **LEGAL DISCLAIMER FOR SYSTEM:** This framework is a legal awareness tool for
> educational purposes only. It does not constitute legal advice and should not
> be used as a substitute for consultation with a qualified legal practitioner.
> All outputs must include this disclaimer prominently.

---

*Build Guide Version 1.0 | Legal IPC-RAG | March 2026*
