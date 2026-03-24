# Legal IPC-RAG Framework ⚖️

A state-of-the-art **Retrieval-Augmented Generation (RAG)** framework designed for **IPC Misuse Detection** and **Citizen-Centric Legal Awareness** in First Information Report (FIR) documents.

This project implements a novel **IPC Contextual Alignment Module (IPC-CAM)** that evaluates semantic consistency between FIR narrative facts and statutory legal elements using a 3-stage pipeline:
1. Semantic Similarity
2. Natural Language Inference (NLI)
3. LLM Reasoning

## 🚀 System Architecture
- **Knowledge Base:** 474 parsed and enriched Indian Penal Code sections.
- **Retrieval:** Hybrid pipeline using BM25 and Dense Vector representations (`GIST-large-Embedding-v0`).
- **Re-ranking:** Cross-encoder evaluation (`bge-reranker-large`).
- **Generative Engine:** Fine-tuned `Llama-3.2-3B` for legal explanations and rationale generation.
- **Frontend:** Professional Streamlit dashboard for end-to-end FIR auditing.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- **OS:** Linux/Windows (WSL recommended)
- **Python:** `3.10` or `3.11`
- **GPU:** NVIDIA GPU with at least 6GB VRAM (Required for local LLM inference and fine-tuning). CUDA `11.8+` recommended.

### 2. Clone the Repository
```bash
git clone https://github.com/SKrishna-7/legal-query-rag-ipc.git
cd legal-query-rag-ipc
```

### 3. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📥 Downloading Required Models

This framework relies on several advanced NLP models. You must download them to your local cache before running the application.

### 1. HuggingFace Authentication
Some models (like LLaMA-3.2) require authentication. 
1. Get an access token from [HuggingFace Settings](https://huggingface.co/settings/tokens).
2. Login via the terminal:
```bash
huggingface-cli login
```

### 2. Download the Models
Run the following commands to pre-download the specific weights used by the pipeline:

**Embedding Model:**
```bash
huggingface-cli download avsolatorio/GIST-large-Embedding-v0
```

**Reranker Model:**
```bash
huggingface-cli download BAAI/bge-reranker-large
```

**NLI Entailment Model (for IPC-CAM):**
```bash
huggingface-cli download cross-encoder/nli-deberta-v3-small
```

**Generative LLM (Fine-tuning Base):**
```bash
huggingface-cli download meta-llama/Llama-3.2-3B
```

*(Note: Ensure you have accepted the user agreement for LLaMA on the HuggingFace website before downloading).*

---

## 🔑 Environment Variables
Create a `.env` file in the root directory and add any required API keys. If you are using the Groq API for the Streamlit chat interface, add it here:

```env
GROQ_API_KEY="your_api_key_here"
```

---

## 💻 Running the Application

### 1. Fine-Tuning the IPC LLM
If you want to train the model on the specialized synthetic IPC dataset, run the generative fine-tuning script:
```bash
python src/generative/finetune_ipc_llm.py
```

### 2. Launching the Dashboard
To start the professional user interface where you can upload FIR PDFs and view the legal rationale audit:
```bash
streamlit run app.py
```
*The app will be available locally at `http://localhost:8501`*

---

## 📁 Repository Structure
* `src/` - Core logic containing the IPC-CAM module, preprocessors, and RAG pipelines.
* `data/` - Metadata, synthetic JSON outputs, and evaluation metrics. *(Note: Raw training data/PDFs are excluded to maintain repository health).*
* `app.py` - The main Streamlit dashboard application.
* `paper/` - Research paper drafts outlining the theoretical foundation of this work.

---
*Developed for Legal NLP Research and Citizen Awareness. Not a substitute for professional legal counsel.*