"""
LLM interface supporting multiple models (API and local).
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    name: str
    model_type: str  # 'openai', 'anthropic', 'local', 'huggingface'
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    top_p: float = 0.9


class BaseLLM(ABC):
    """Base class for all LLM implementations."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    def __str__(self):
        return f"{self.config.model_type}:{self.config.model_name}"


class OpenAILLM(BaseLLM):
    """OpenAI API implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.OpenAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("openai package required for OpenAI models")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p)
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"Error: {str(e)}"


class AnthropicLLM(BaseLLM):
    """Anthropic API implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("anthropic package required for Anthropic models")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return f"Error: {str(e)}"


class HuggingFaceLLM(BaseLLM):
    """HuggingFace local model implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except ImportError:
            raise ImportError("transformers and torch packages required for HuggingFace models")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                    temperature=kwargs.get('temperature', self.config.temperature),
                    top_p=kwargs.get('top_p', self.config.top_p),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            return generated_text.strip()
            
        except Exception as e:
            print(f"HuggingFace generation error: {e}")
            return f"Error: {str(e)}"


class OllamaLLM(BaseLLM):
    """Ollama local model implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import requests
            self.base_url = config.base_url or "http://localhost:11434"
            self.session = requests.Session()
        except ImportError:
            raise ImportError("requests package required for Ollama models")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get('temperature', self.config.temperature),
                        "top_p": kwargs.get('top_p', self.config.top_p),
                        "num_predict": kwargs.get('max_tokens', self.config.max_tokens)
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Ollama API error: {e}")
            return f"Error: {str(e)}"


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(config: LLMConfig) -> BaseLLM:
        """Create LLM instance based on config."""
        if config.model_type == "openai":
            return OpenAILLM(config)
        elif config.model_type == "anthropic":
            return AnthropicLLM(config)
        elif config.model_type == "huggingface":
            return HuggingFaceLLM(config)
        elif config.model_type == "ollama":
            return OllamaLLM(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")


class LLMBenchmark:
    """Benchmark multiple LLMs on the same task."""
    
    def __init__(self, configs: List[LLMConfig]):
        self.llms = [LLMFactory.create_llm(config) for config in configs]
    
    def run_benchmark(self, prompts: List[str], **generation_kwargs) -> Dict[str, List[str]]:
        """Run benchmark across all LLMs."""
        results = {}
        
        for llm in self.llms:
            print(f"Running benchmark on {llm}")
            llm_results = []
            
            for i, prompt in enumerate(prompts):
                start_time = time.time()
                try:
                    output = llm.generate(prompt, **generation_kwargs)
                    generation_time = time.time() - start_time
                    llm_results.append({
                        'output': output,
                        'generation_time': generation_time,
                        'prompt_index': i
                    })
                except Exception as e:
                    llm_results.append({
                        'output': f"Error: {str(e)}",
                        'generation_time': -1,
                        'prompt_index': i
                    })
                
                print(f"  Completed prompt {i+1}/{len(prompts)}")
            
            results[str(llm)] = llm_results
        
        return results


# Predefined model configurations
PREDEFINED_CONFIGS = {
    'gpt-3.5-turbo': LLMConfig(
        name='gpt-3.5-turbo',
        model_type='openai',
        model_name='gpt-3.5-turbo',
        api_key=os.getenv('OPENAI_API_KEY')
    ),
    'gpt-4': LLMConfig(
        name='gpt-4',
        model_type='openai',
        model_name='gpt-4',
        api_key=os.getenv('OPENAI_API_KEY')
    ),
    'claude-3-sonnet': LLMConfig(
        name='claude-3-sonnet',
        model_type='anthropic',
        model_name='claude-3-sonnet-20240229',
        api_key=os.getenv('ANTHROPIC_API_KEY')
    ),
    'llama-2-7b': LLMConfig(
        name='llama-2-7b',
        model_type='huggingface',
        model_name='meta-llama/Llama-2-7b-chat-hf'
    ),
    'mistral-7b': LLMConfig(
        name='mistral-7b',
        model_type='huggingface',
        model_name='mistralai/Mistral-7B-Instruct-v0.1'
    ),
    'ollama-llama2': LLMConfig(
        name='ollama-llama2',
        model_type='ollama',
        model_name='llama2'
    )
}