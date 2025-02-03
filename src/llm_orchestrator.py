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
            n_threads=4,
            n_gpu_layers=-1
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
            self.prompt_template.format(
                context=context,
                question=query
            ),
            max_tokens=512,
            temperature=0.7,
            stop=["[/INST]", "</s>"]
        )
        
        generated_text = response["choices"][0]["text"]
        self.memory.store_memory(user_id, query, generated_text)

        return generated_text