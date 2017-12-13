#패키지 불러옴
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

#사용할 단어 입력
input_words = ['writing', 'calves', 'be', 'branded', 'horse', 'randomize', 'possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code']

#스테머 객체 생성
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english')

#출력을 위한 문자열 포맷 정의
stemmer_names = ['PORTER', 'LANCASTER', 'SNOWBALL']
formatted_text = '{:>16}' * (len(stemmer_names)+1)
print('\n', formatted_text.format('INPUT WORD', *stemmer_names), '\n', '='*68)

#입력 단어별로 어간 추출해 출력
for word in input_words:
    output=[word, porter.stem(word), lancaster.stem(word), snowball.stem(word)]
    print(formatted_text.format(*output)) 
