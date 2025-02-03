from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .memory_manager import MemorySystem
from .web_search import WebSearch
import torch
load_dotenv()

class LLMOrchestrator:
    def __init__(self):
        self.memory = MemorySystem()
        self.web_search = WebSearch()
        
        # Load local LLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            "models/mistral-7b-vietnamese",
            use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "models/mistral-7b-vietnamese",
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512
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
        response = self.pipe(
            prompt,
            temperature=0.7,
            do_sample=True
        )[0]['generated_text']
        
        # 5. Update memory
        self.memory.store_memory(user_id, query, response)
        
        return response.split("[/INST]")[-1].strip()