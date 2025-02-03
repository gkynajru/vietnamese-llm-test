from src.llm_orchestrator import LLMOrchestrator
import time

def main():
    assistant = LLMOrchestrator()
    
    while True:
        query = input("\nBạn: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        start_time = time.time()
        response = assistant.generate_response(query)
        latency = time.time() - start_time
        
        print(f"\nTrợ lý ({latency:.2f}s): {response}")

if __name__ == "__main__":
    main()