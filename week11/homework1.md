## **1. 연구 배경**  
선행 연구에서 nltk 패키지를 이용하여 IMDb 영화 리뷰 데이터 감정 분석을 시행했을 때, 실제 리뷰 내용과 예측 결과가 상이하게 도출되는 경우가 많았다. ‘A Clockwork Orange’의 경우 리뷰의 실제 sentiment와 예측된 sentiment가 불일치하는 경우가 20개 중 6개였고, ‘Into the Wild’의 경우 리뷰의 실제 sentiment와 예측된 sentiment가 불일치하는 경우가 20개 중 3개였다. 특히 실제로는 부정적 리뷰이나, 긍정적 리뷰로 예측되는 결과가 많았으며, 실제 내용과 불일치하는 예측 결과임에도 불구하고 probability가 최소 0.62, 대부분 0.90 이상으로 높은 수치를 보이는 결과가 도출되었다. 선행 연구의 부정확한 예측 결과를 바탕으로, 분석의 정확도를 낮추고, 결과 예측에 영향을 미치는 요인을 알아보고자 한다.

## **2. 선행 연구**  
선행 연구의 분석 데이터로는 두 영화 ‘A Clockwork Orange (Stanley Kubrick, 1971)’, ‘Into the Wild (Sean penn, 2007)’의 리뷰를 이용하였다. 두 영화의 리뷰 중 긍정적 리뷰, 부정적 리뷰를 각각 10개씩 영화당 20개의 리뷰를 선정하였다. 리뷰 선정 기준은 Loved It/Hated It을 필터로 적용했을 때 상위 노출되는 순서이다. 분석 결과, 위에서 말한 바와 같이 A Clock Orange의 경우 리뷰 실제 내용과 예측된 sentiment가 불일치 하는 경우가 20개 중 6개 존재하였고, Into the Wild의 경우에는 20개 중 3개의 예측 sentiment가 실제 리뷰 내용과 불일치 하였다. 또한, 두 영화 모두 긍정적인 내용의 리뷰가 부정적 내용의 리뷰로 예측되는 경우보다, 부정적인 내용의 리뷰가 긍정적 내용의 리뷰로 예측되는 경우가 월등히 더 많았다.

## **3. 연구 가설**  
선행 연구의 분석 결과를 바탕으로, 분석 결과에 영향을 미치는 요인을 구두점, 부정어, 복문의 여부 세가지로 설정한다. 그리고 세가지 요인과 예측 결과의 상관관계를  알아보기 위해 다음과 같은 3가지 가설을 설정하고자 한다.

- 연구 가설 1. 구두점(Punctuation)이 많을수록 예측의 정확도가 떨어진다.

- 연구 가설 2. 부정어 ( not, never, no 등)을 많이 사용할 수록 예측의 정확도가 떨어진다.

- 연구 가설 3. 같은 의미의 리뷰일 경우, 하나의 문장으로 문장 안에 단어를 나열하는 경우보다 복수의 문장으로 내용을 입력하는 경우 probability가 높아진다.

## **4. 연구방법**  
분석 데이터로는 선행 연구에서 이용한 분석 데이터를 이용하되, 실제 리뷰 내용과 예측 결과가 상이했던 총 9가지 리뷰를 중심적으로 이용했다. 연구  가설 1의 타당성을 판단하기 위하여, 우선 선행 연구의 분석 데이터의 구두점을 제거한 후 재분석을 실시했다. 또한 앞에서 언급한 부정확한 예측 결과가 도출되었던 9가지 리뷰들만 분석 데이터로 설정한 후 다시 한 번 분석하였다.
또한, 연구 가설 2의 타당성을 판단하기 위해서 위의 9가지 리뷰 중 부정어가 포함된 분석 데이터와, 문장에 부정어가 있는 새로운 데이터를 취합하여 분석을 실시하였다.
마지막으로 연구 가설 3의 타당성을 판단하기 위해, 같은 의미의 문장을 단일 문장, 복수의 문장으로 구성한 데이터를 분석하였다. 복수 문장의 수는 2-5개까지로 설정하였다.

## **5. 연구결과**  
- 연구 가설 1. 구두점(Punctuation)이 많을수록 예측의 정확도가 떨어진다.  [코드](https://github.com/ttthy1/2017sejongAI/blob/master/week11/sentiment_analyzer_punc.py)  
    
  데이터의 구두점을 제거하여 분석한 결과, input 데이터의 sentiment와 predicted sentiment가 일치하는 결과가 도출되는 경우의 수가 1증가 하였다. 해당 데이터에서 제거한 구두점은 ‘!!’, ‘’’, ‘,’, ‘.’ 4가지 이다. 또한 실제 내용과 예측 결과의 불일치가 그대로 유지되는 경우에도 예측의 Probability 수치가 낮아지는 경우가 많았다. 따라서 구두점은 예측의 정확도에 영향을 미치는 요인이다. 즉, 예측의 정확도와 구두점 사이에는 상관관계가 존재한다.  
  
    >Review: This movie sucks!! From the clothes, to the retarded mix of English and Russian to the theatrical way the violent scenes are shot. The theatrical shooting of the violence, makes it impossible to take it seriously. I've seen more graphic violence in modern TV-series than in this over-rated crap. If you feel like you HAVE to watch this movie because it's considered a classic, don't put yourself through the torture. It's classic BS and it should be avoided completely.  
    >Predicted sentiment: Positive  
    >Probability: 0.95  
  
    >Review: This movie sucks From the clothes to the retarded mix of English and Russian to the theatrical way the violent scenes are shot The theatrical shooting of the violence makes it impossible to take it seriously I have seen more graphic violence in modern TV series than in this over rated crap If you feel like you HAVE to watch this movie because it is considered a classic do not put yourself through the torture It is classic BS and it should be avoided completely  
    >Predicted sentiment: Negative  
    >Probability: 0.62

- 연구 가설 2. 부정어 ( not, never, no 등)을 많이 사용할 수록 예측의 정확도가 떨어진다.  [코드](https://github.com/ttthy1/2017sejongAI/blob/master/week11/sentiment_analyzer_neg.py)  
   
  부정어가 포함된 선행 연구의 데이터와 리뷰에 부정어가 사용된 새로운 데이터를 취합하여 분석한 결과, 부정어를 사용하는 경우 예측의 정확도가 떨어졌다. 특히 be동사의 부정형, 혹은 형용사 앞에 부정어를 사용하는 경우 실제 리뷰의 sentiment와는 다르게 sentiment를 예측하는 경우가 많았다. 즉, 부정어 또한 예측의 정확도에 영향을 미치는 요인이며, 예측의 정확도와 부정어 사이에는 상관관계가 존재한다.  
    
    >Review: There is no meaningless dialog; not a single extraneous character.  
    >Predicted sentiment: Negative  
    >Probability: 0.86  
      
    >Review: I don't see any point to the vast majority of this film.  
    >Predicted sentiment: Positive  
    >Probability: 0.73  
      
    >Review: there is no justice in the world and that's acceptable for humankind.  
    >Predicted sentiment: Positive  
    >Probability: 0.58  
    
    >Review: Not terrifying, The main problem is that I don't see any message in this movie, nor any pleasant feature. It is not funny, it is not horrifying or realistic enough, there is no real character development, no interesting statements about anything. It is an empty, overrated mess. It is basically a shock-parade with old instrumental background music.  
    >Predicted sentiment: Positive  
    >Probability: 0.86
    
    >Review: nothing really interesting or compelling ever happens in this.  
    >Predicted sentiment: Positive  
    >Probability: 0.58  
      
    >Review: It is terrifying  
    >Predicted sentiment: Positive  
    >Probability: 0.7 
  
    >Review: It is not terrifying  
    >Predicted sentiment: Positive  
    >Probability: 0.7
      
- 연구 가설 3. 같은 의미의 리뷰일 경우, 하나의 문장으로 문장 안에 단어를 나열하는 경우보다 복수의 문장으로 내용을 입력하는 경우 Probability가 높아진다.  [코드]
  (https://github.com/ttthy1/2017sejongAI/blob/master/week11/sentiment_analyzer_nos.py)  
  같은 의미의 리뷰를 단일 문장으로 축약한 리뷰와, 복수의 문장을 나열하는 리뷰를 비교한 결과, 복수의 문장을 나열할수록 Probability가 높아졌다. 문장의 수가 많아질수록, 예측 결과의 Probability가 증가하는 것으로 나타났다. 따라서 문장의 수와 Probability 사이에는 양의 상관관계가 있는 것으로 예측할 수 있다.  
  
    >Review: This is a great, awesome, funny, beautiful,perfect movie.  
    >Predicted sentiment: Positive  
    >Probability: 0.5  
  
    >Review: This is a great, awesome, funny movie. This is a beautiful, perfect movie.  
    >Predicted sentiment: Positive  
    >Probability: 0.7  
  
    >Review: This is a great, awesome movie. This is a funny movie. This is a beautiful, perfect movie.  
    >Predicted sentiment: Positive  
    >Probability: 0.84  
  
    >Review: This is a great, awesome movie. This is a funny movie. This is a beautiful movie. This is a perfect movie.  
    >Predicted sentiment: Positive  
    >Probability: 0.89  
  
    >Review: This is a great movie. This is a awesome movie. This is a funny movie. This is a beautiful movie. This is a perfect movie.  
    >Predicted sentiment: Positive  
    >Probability: 0.93  

## **6. 결론 및 한계점**  
세 가지의 연구 가설의 타당성을 분석해 본 결과, 예측의 정확도와 구두점,부정어 사이에는 각각 상관관계가 존재하는 것으로 예측해 볼 수 있으며, 문장의 수와 Probability 수치에는 양의 상관관계가 있는 것으로 예측해 볼 수 있다. 따라서, 리뷰 내용에서 구두점과 부정어 사용이 낮을수록 더 정확한 예측이 가능하며, 문장의 수가 복문일 경우 Probability가 높은 결과를 도출하는 것이 가능하다.
그러나 이 연구에는 몇가지 한계점이 존재한다. 첫째로, 분석 데이터의 수가 적어 표본오차가 크다는 한계점이 존재한다. 둘째로, 분석 데이터의 선정 기준이 다소 자의적이어서, 표본으로 선정된 분석 데이터들이 모집단을 대표하는 대표성이 낮을 수 있다는 한계점이 존재한다. 마지막으로 단순히 구두점을 없애거나, 간략하게 문장에 부정어를 삽입하거나 생략하는 경우 분석 데이터의 의미 자체가 변화하여 정확한 예측이 어렵다는 한계점이 있다.
