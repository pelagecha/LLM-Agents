import requests
from bs4 import BeautifulSoup

url = "https://www.engvid.com/english-resource/50-fun-english-riddles/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
riddles = []
for li in soup.find_all('li'):
    question = li.get_text(strip=True).split('show answer')[0] 
    answer_tag = li.find('span', class_='riddle_answer') 
    
    if answer_tag:
        answer = answer_tag.get_text(strip=True)
        riddles.append((question, answer))

with open('riddles.txt', 'w') as f:
    for question, _ in riddles:
        question, answer = question.split("?")
        f.write(f"{question}? <SPLIT> {answer}\n")

print("Riddles have been saved to riddle.txt")
