
#패키지 호출
import numpy as np
from nltk.corpus import brown

#입력 테스트를 단어 묶음으로 나눔
#단어 묶음별로 N개의 단어 포함
def chunker(input_data, N):
 input_words = input_data.split(' ')
 output = []
 
#매개변수에 따라 텍스트를 N 단어 묶음으로 변환하는 함수 정의 리스트 반환
 cur_chunk = []
 count = 0
 for word in input_words:
 cur_chunk.append(word)
 count += 1
 if count == N:
 output.append(' '.join(cur_chunk))
 count, cur_chunk = 0, []
 output.append(' '.join(cur_chunk))
 return output

#메인 함수 정의, 브라운 말뭉치를 입력데이터로 사용
#12,000개 단어를 읽어서 입력 데이터로 사용
if __name__=='__main__':   
 # 브라운 코퍼스에서 12,000개 단어를 읽어옴
 input_data = ' '.join(brown.words()[:12000])
 #묶음에 포함될 단어 수정의
 chunk_size = 700
#입력 테스트를 단어 묶으로 나누고 결과 표시
 chunks = chunker(input_data, chunk_size)
 print('\nNumber of text chunks =', len(chunks), '\n')
 for i, chunk in enumerate(chunks):
 print('Chunk', i+1, '==>', chunk[:50]) 
