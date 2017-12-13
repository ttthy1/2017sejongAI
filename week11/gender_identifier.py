# 입력 단어에서 마지막 N개의 문자를 추출하는 함수 정의
import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names

# 입력 단어에서 마지막 N개의 문자를 추출하는 함수 정의
# 입력단어에서 마지막 N개 글자를 추출해 자질로 사용
def extract_features(word, N=2):
    last_n_letters = word[-N:]
    return {'feature': last_n_letters.lower()}
#메인함수 정의, 사이킷런 패키지에서 학습데이터 추출 (이 데이터에는 남성과 여성으로 분류된 이름이 포함되어 있음)
if __name__=='__main__':
    # NLTK의 이름 정보를 이용해 학습셋 만듬
    male_list = [(name, 'male') for name in names.words('male.txt')]
    female_list = [(name, 'female') for name in names.words('female.txt')]
    data = (male_list + female_list)
# 난수 발생기에 시드 값 전달
random.seed(5)

# 데이터 뒤섞기
random.shuffle(data)

# 테스트 데이터 만들기
input_names = ['Alexander', 'Danielle', 'David', 'Cheryl']

# 학습셋과 테스트셋 비율 정하기
num_train = int(0.8 * len(data))

# 글자 수별로 성능 비교
for i in range(1, 6):
    print('\nNumber of end letters:', i)
    features = [(extract_features(n, i), gender) for (n, gender) in data]

#데이터를 학습셋과 데스트셋으로 분리
train_data, test_data = features[:num_train], features[num_train:]

#학습데이터를 사용해 나이브 베이즈 분류기 생성
classifier = NaiveBayesClassifier.train(train_data)

# 분류기 정확도 계산
accuracy = round(100 * nltk_accuracy(classifier, test_data), 2)
print('Accuracy = ' + str(accuracy) + '%')

# 학습한 분류기로 결과 예측
for name in input_names:
    print(name, '==>', classifier.classify(extract_features(name, i)))
