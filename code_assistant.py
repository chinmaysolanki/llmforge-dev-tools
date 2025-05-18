import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

class CodeAssistant:
    def __init__(
        self,
        model_name: str = "microsoft/CodeGPT-small-py",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2048,
        temperature: float = 0.7
    ):
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        
        logging.info(f"Loading model {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Load programming language specific prompts
        self.language_prompts = self._load_language_prompts()
        
    def _load_language_prompts(self) -> Dict[str, str]:
        """Load language-specific system prompts."""
        return {
            "python": "You are a Python expert. Provide clear, efficient, and well-documented code.",
            "javascript": "You are a JavaScript expert. Write modern, clean, and maintainable code.",
            "typescript": "You are a TypeScript expert. Focus on type safety and best practices.",
            "java": "You are a Java expert. Follow object-oriented principles and design patterns.",
            "cpp": "You are a C++ expert. Write efficient and memory-safe code.",
            "rust": "You are a Rust expert. Focus on safety, performance, and zero-cost abstractions."
        }
    
    def generate_code(
        self,
        prompt: str,
        language: str = "python",
        context: Optional[List[str]] = None
    ) -> str:
        """
        Generate code based on the prompt and context.
        
        Args:
            prompt: The user's request or question
            language: The target programming language
            context: Optional list of relevant code snippets or context
            
        Returns:
            Generated code as a string
        """
        system_prompt = self.language_prompts.get(language.lower(), "")
        
        # Construct the full prompt with context
        full_prompt = f"{system_prompt}\n\n"
        if context:
            full_prompt += "Context:\n" + "\n".join(context) + "\n\n"
        full_prompt += f"Request: {prompt}\n\nResponse:"
        
        # Tokenize and generate
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Response:")[-1].strip()
    
    def analyze_code(self, code: str, language: str = "python") -> Dict:
        """
        Analyze code for potential improvements, bugs, or security issues.
        
        Args:
            code: The code to analyze
            language: The programming language of the code
            
        Returns:
            Dictionary containing analysis results
        """
        prompt = f"Analyze this {language} code for potential issues, improvements, and best practices:\n\n{code}"
        analysis = self.generate_code(prompt, language)
        
        return {
            "analysis": analysis,
            "language": language,
            "timestamp": str(datetime.now())
        }
    
    def explain_code(self, code: str, language: str = "python") -> str:
        """
        Generate a detailed explanation of the provided code.
        
        Args:
            code: The code to explain
            language: The programming language of the code
            
        Returns:
            Detailed explanation of the code
        """
        prompt = f"Explain this {language} code in detail, including its purpose, how it works, and any important concepts:\n\n{code}"
        return self.generate_code(prompt, language)

if __name__ == "__main__":
    # Example usage
    assistant = CodeAssistant()
    
    # Example code generation
    prompt = "Create a function to find the nth Fibonacci number"
    code = assistant.generate_code(prompt, language="python")
    print("Generated Code:")
    print(code)
    
    # Example code analysis
    analysis = assistant.analyze_code(code)
    print("\nCode Analysis:")
    print(json.dumps(analysis, indent=2)) 