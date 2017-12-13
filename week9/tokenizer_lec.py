#패키지 호출
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer

#입력테스트 정의
input_text = "Do you know how tokenization works? It's actually quite interesting! Let's analyze a couple of sentences and figure it out."

#입력 텍스트를 문장 토큰으로 나눔
print("\nSentence Tokenizer:")
print(sent_tokenize(input_text))

#입력 텍스트를 단어 토큰으로 나눔
print("\nWord tokenizer : ")
print(word_tokenize(input_text))

#Word punct tokenizer사용해 입력 텍스트를 단어 토큰으로 나눔
print("\nWord Punct Tokenizer : ")
print(WordPunctTokenizer().tokenize(input_text))
