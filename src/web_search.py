from googlesearch import search
import requests
from bs4 import BeautifulSoup

class WebSearch:
    def search(self, query, num_results=3):
        try:
            # Get Vietnamese results
            sites = list(search(
                query, 
                lang="vi",
                num_results=num_results
            ))
            
            # Simple content extraction
            contents = []
            for url in sites:
                page = requests.get(url, timeout=5)
                soup = BeautifulSoup(page.content, 'html.parser')
                contents.append(soup.get_text()[:500])
                
            return " ".join(contents)
        except:
            return "Không tìm thấy thông tin từ web."