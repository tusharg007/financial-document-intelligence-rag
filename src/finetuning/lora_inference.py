"""
Fine-Tuned LLM inference module.

Loads a LoRA-adapted model and provides the same interface as HuggingFaceLLM
so it can be a drop-in replacement inside the RAG pipeline.

Usage:
    from src.finetuning.lora_inference import FineTunedLLM
    
    llm = FineTunedLLM(
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_adapter="./lora_findoc"  # or "your-hf-username/findoc-lora"
    )
    answer = llm.generate("Your prompt here")
"""

import torch
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("lora_inference")


class FineTunedLLM:
    """
    Drop-in replacement for HuggingFaceLLM that uses a local LoRA-adapted model.
    
    Compatible with the .generate(prompt, max_tokens) interface used in rag_agent.py.
    """
    
    def __init__(
        self,
        base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_adapter: str = "./lora_findoc",
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        max_memory: Optional[dict] = None,
    ):
        """
        Load the base model with LoRA adapter merged.
        
        Args:
            base_model: HuggingFace model ID for the base model
            lora_adapter: Path or HF repo ID for the LoRA adapter
            device: Device to load on ("cuda", "cpu", or None for auto)
            load_in_4bit: Use 4-bit quantization for memory efficiency
            max_memory: GPU memory limit dict, e.g. {0: "6GiB"}
        """
        self.base_model_name = base_model
        self.lora_adapter_path = lora_adapter
        self._model = None
        self._tokenizer = None
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._max_memory = max_memory
    
    def _ensure_loaded(self):
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        logger.info(f"Loading base model: {self.base_model_name}")
        logger.info(f"Loading LoRA adapter: {self.lora_adapter_path}")
        
        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.lora_adapter_path,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Quantization config
        bnb_config = None
        if self._load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Device map
        device_map = "auto"
        if self._device:
            device_map = {"": self._device}
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            max_memory=self._max_memory,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Apply LoRA adapter
        self._model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)
        self._model.eval()
        
        logger.info("✅ Fine-tuned model loaded successfully")
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Generate text using the fine-tuned model.
        
        This method signature matches HuggingFaceLLM.generate() exactly,
        so it can be used as a drop-in replacement in RAGPipeline.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum new tokens to generate
            
        Returns:
            Generated text string
        """
        try:
            self._ensure_loaded()
            
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self._model.device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.15,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            
            # Decode only the new tokens (skip the input)
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Fine-tuned model generation failed: {e}")
            return ""
    
    def is_available(self) -> bool:
        """Check if the model can be loaded."""
        try:
            from pathlib import Path
            # Check local path
            if Path(self.lora_adapter_path).exists():
                return True
            # Check HF Hub
            from huggingface_hub import model_info
            model_info(self.lora_adapter_path)
            return True
        except Exception:
            return False
