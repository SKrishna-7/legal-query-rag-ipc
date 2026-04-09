import os
import json
import time
from pathlib import Path
from groq import Groq
from tqdm import tqdm

# Configure paths and API
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
INPUT_PATH = "data/processed/ipc_sections/ipc_sections_list.json"
OUTPUT_PATH = "data/processed/ipc_sections/ipc_complete.json"
INDIVIDUAL_DIR = "data/processed/ipc_sections/individual"

# Ensure directories exist
Path(INDIVIDUAL_DIR).mkdir(parents=True, exist_ok=True)

client = Groq(api_key=GROQ_API_KEY)

def extract_structured_data(section, max_retries=5):
    """
    Use Groq API to extract structured data from an IPC section.
    Implements exponential backoff for rate limiting.
    """
    # Truncate text if too long (max 10,000 chars for safety)
    safe_text = section['full_text'][:10000]
    
    prompt = f"""
    You are an expert Indian Legal Advisor. Analyze the following IPC section text and extract its legal elements into the exact JSON format specified.
    CRITICAL: You must use exactly the section number, title, chapter, chapter_title, and full_text provided below.

    SECTION NUMBER: {section['section_number']}
    TITLE: {section['title']}
    CHAPTER: {section['chapter']}
    CHAPTER TITLE: {section['chapter_title']}

    SECTION TEXT:
    {safe_text}

    JSON FORMAT:
    {{
      "section_number": "{section['section_number']}",
      "title": "{section['title']}",
      "chapter": "{section['chapter']}",
      "chapter_title": "{section['chapter_title']}",
      "full_text": "...",
      "punishment": "Summary of punishment mentioned in the text",
      "essential_ingredients": [
        "First requirement to satisfy this section",
        "Second requirement...",
        "..."
      ],
      "related_sections": ["Other section numbers related to this one"],
      "cognizable": true/false (based on Indian law for this section),
      "bailable": true/false (based on Indian law for this section),
      "triable_by": "Name of the court (e.g., Magistrate of first class, Court of Session)",
      "compoundable": true/false,
      "keywords": ["keyword1", "keyword2"],
      "mens_rea_required": true/false,
      "actus_reus": "Summary of the physical act required",
      "maximum_punishment": "Maximum jail term or death",
      "minimum_punishment": "Minimum jail term or none"
    }}

    Return ONLY the valid JSON object. No preamble or explanation.
    """

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                temperature=0.1
            )
            data = json.loads(response.choices[0].message.content)
            
            # Enforce original section details to prevent hallucination
            data['section_number'] = section['section_number']
            data['title'] = section['title']
            data['chapter'] = section['chapter']
            data['chapter_title'] = section['chapter_title']
            if 'full_text' not in data or len(data['full_text']) < 10:
                data['full_text'] = section['full_text']
            
            return data
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                wait_time = (2 ** attempt) + 5  # Exponential backoff: 6s, 7s, 9s, 13s, 21s
                print(f"Rate limit hit for Section {section['section_number']}. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Error processing Section {section['section_number']}: {e}")
                return None
    
    print(f"Failed to process Section {section['section_number']} after {max_retries} attempts.")
    raise Exception("Rate limit exceeded persistently. Stopping script.")

def main():
    if not Path(INPUT_PATH).exists():
        print(f"Error: {INPUT_PATH} not found.")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        sections = json.load(f)

    # Load existing to avoid re-processing
    complete_data = {}
    if Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing_list = json.load(f)
            # Filter out invalid records with empty or null section numbers
            for s in existing_list:
                sec_num = s.get('section_number')
                # Check if sec_num is a valid string/number that's not empty and matches original list
                if sec_num and str(sec_num).strip() != "" and str(sec_num) != "None" and str(sec_num) != "Unknown" and str(sec_num) != "UNKNOWN" and "specified" not in str(sec_num).lower() and "mentioned" not in str(sec_num).lower():
                    complete_data[str(sec_num)] = s

    print(f"Starting enrichment of {len(sections)} sections...")
    print(f"Already completed valid sections: {len(complete_data)}")
    
    # Process each section
    for section in tqdm(sections):
        sec_num = section.get('section_number')
        if not sec_num or str(sec_num).strip() == "":
            continue # Skip invalid sections in source list
            
        sec_num = str(sec_num)
        
        # Skip if already processed
        if sec_num in complete_data:
            continue
            
        enriched = extract_structured_data(section)
        if enriched:
            complete_data[sec_num] = enriched
            
            # Save individual file
            with open(f"{INDIVIDUAL_DIR}/section_{sec_num}.json", "w", encoding="utf-8") as f:
                json.dump(enriched, f, indent=2, ensure_ascii=False)
            
            # Save periodically to avoid data loss
            if len(complete_data) % 5 == 0:
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(list(complete_data.values()), f, indent=2, ensure_ascii=False)
        
        # Base rate limit protection
        time.sleep(3) # Increased base sleep to 3 seconds to avoid hitting limits quickly

    # Final save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(list(complete_data.values()), f, indent=2, ensure_ascii=False)
    
    print(f"Finished! Processed {len(complete_data)} sections in total.")

if __name__ == "__main__":
    main()
