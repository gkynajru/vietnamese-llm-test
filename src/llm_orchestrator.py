from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from llama_cpp import Llama
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .memory_manager import MemorySystem
from .web_search import WebSearch
import torch
load_dotenv()

class LLMOrchestrator:
    def __init__(self):
        self.memory = MemorySystem()
        self.web_search = WebSearch()
        
        # Load local LLM
        self.model = Llama(
            model_path="models/mistral-7b-vietnamese/ggml-vistral-7B-chat-q4_1.gguf",
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=35,
            verbose=False,  # Reduce logging
            offload_kqv=True,  # Enable KQV offloading
            seed=-1,  # Random seed for reproducibility
            use_mmap=True,  # Use memory mapping
            use_mlock=False  # Don't lock memory
        )

        # System prompt template
        self.prompt_template = """<s>[INST] 
        Bạn là trợ lý ảo tiếng Việt. Sử dụng thông tin sau:
        {context}
        
    Câu hỏi: {question} 
    Trả lời bằng tiếng Việt: [/INST]"""

    def generate_response(self, query, user_id="default"):
        # 1. Retrieve from memory
        context = self.memory.retrieve_context(query, user_id)
        
        # 2. If no relevant memory, search web
        if not context:
            web_results = self.web_search.search(query)
            context = f"Thông tin từ web: {web_results[:1000]}"
            
        # 3. Format prompt
        prompt = self.prompt_template.format(
            context=context,
            question=query
        )
        
        # 4. Generate response
        response = self.model(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.1,
            top_k=40,
            echo=False,
            stop=["[/INST]", "</s>"],
            mirostat_mode=2,  # Enable Mirostat sampling
            mirostat_tau=5.0,
            mirostat_eta=0.1
        )
        
        generated_text = response["choices"][0]["text"]
        self.memory.store_memory(user_id, query, generated_text)

        return generated_text