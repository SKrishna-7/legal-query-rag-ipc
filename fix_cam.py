import re

with open('src/ipc_cam/ipc_cam.py', 'r') as f:
    content = f.read()

# Using regex to replace the entire compute_ingredient_satisfaction_score method
pattern = re.compile(r'    def compute_ingredient_satisfaction_score.*?return IngredientScore\([^)]+\)', re.DOTALL)

replacement = """    def compute_ingredient_satisfaction_score(self,
                                             ingredient: str,
                                             fir_narrative: str) -> IngredientScore:
        \"\"\"Evaluate if an ingredient is satisfied by the FIR narrative using LLM analysis on the full text.\"\"\"
        
        llm_verdict = "NOT_SATISFIED"
        confidence = 0.0
        reasoning = ""
        evidence_sentences = []
        
        prompt = f\"\"\"
        You are a strict Indian Legal Expert. Analyze the following FIR narrative to determine if the specific legal ingredient requirement is satisfied.
        
        CRITICAL INSTRUCTION: Read the ENTIRE narrative carefully. The evidence might be buried in the 'First information contents' or attached sheets at the end.
        
        Legal Ingredient Requirement: "{ingredient}"
        
        FIR Narrative:
        {fir_narrative[:15000]}
        
        Does the narrative provide explicit facts or evidence that satisfy this legal requirement?
        Answer in the following JSON format:
        {{
            "verdict": "SATISFIED" | "PARTIALLY_SATISFIED" | "NOT_SATISFIED",
            "confidence": (float between 0.0 and 1.0, where 1.0 means perfectly justified),
            "reasoning": "Brief legal reasoning citing specific facts from the text if present."
        }}
        \"\"\"

        if self.use_local and self.local_pipeline:
            try:
                # Local Llama-3.2 generation
                outputs = self.local_pipeline(
                    prompt, 
                    max_new_tokens=200,
                    return_full_text=False
                )
                response_text = outputs[0]['generated_text']
                import re
                import json
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    res = json.loads(json_match.group())
                else:
                    res = {"verdict": "PARTIALLY_SATISFIED", "confidence": 0.5, "reasoning": "Local model returned non-JSON response"}
                
                llm_verdict = res.get("verdict", "NOT_SATISFIED")
                confidence = float(res.get("confidence", 0.0))
                reasoning = res.get("reasoning", "")
            except Exception as e:
                print(f"Local Model Error: {e}")
                llm_verdict = "PARTIALLY_SATISFIED"
                reasoning = f"Error in local inference: {e}"

        elif self.client:
            try:
                import json
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    response_format={"type": "json_object"}
                )
                res = json.loads(chat_completion.choices[0].message.content)
                llm_verdict = res.get("verdict", "NOT_SATISFIED")
                confidence = float(res.get("confidence", 0.0))
                reasoning = res.get("reasoning", "")
                
                # Try to extract the sentence it relied on for evidence tracking
                if "SATISFIED" in llm_verdict:
                     evidence_sentences.append(reasoning)

            except Exception as e:
                print(f"Groq API Error: {e}")
                llm_verdict = "PARTIALLY_SATISFIED"
                reasoning = f"Error calling LLM API: {e}"

        else:
            # Fallback if no LLM
            llm_verdict = "PARTIALLY_SATISFIED"
            confidence = 0.5
            reasoning = "No LLM available for semantic analysis."

        status_map = {
            "SATISFIED": SatisfactionStatus.SATISFIED,
            "PARTIALLY_SATISFIED": SatisfactionStatus.PARTIALLY_SATISFIED,
            "NOT_SATISFIED": SatisfactionStatus.NOT_SATISFIED
        }
        
        return IngredientScore(
            ingredient=ingredient,
            satisfaction_status=status_map.get(llm_verdict, SatisfactionStatus.NOT_SATISFIED),
            confidence_score=confidence,
            extracted_evidence=evidence_sentences,
            reasoning=reasoning
        )"""

new_content = re.sub(pattern, replacement, content)

with open('src/ipc_cam/ipc_cam.py', 'w') as f:
    f.write(new_content)
    
print("Successfully replaced compute_ingredient_satisfaction_score")
