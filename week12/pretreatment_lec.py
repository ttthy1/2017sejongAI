#패키지 호출
import numpy as np
from sklearn import preprocessing

#샘플 데이터 정의
input_data = np.array([[5.1, -2.9, 3.3], [-1.2, 7.8, -6.1],
[3.9, 0.4, 2.1], [7.3, -9.9, -4.5]])

#데이터 이진화 (임계값(threshold)은 2.1로 지정. 즉, 임계값 2.1을 기준으로 1과 0을 구분)
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nBinarized data : \n", data_binarized)

#평균과 표준편차 출력
print("\n BEFORE : ")
print("Mean = ", input_data.mean(axis=0))
print("Std deviation = ", input_data.std(axis=0))

#평균 제거
data_scaed = preprocessing.scale(input_data)
print("\n AFTER : ")
print("Mean = ", data_scaed.mean(axis=0))
print("Std deviation = ", data_scaed.std(axis=0))

#크기 조정
#최솟값 /최댓값 조정
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data : \n", data_scaled_minmax)

#정규화
#데이터 정규화
# L1 정규화 : 각 행의 합이 1이 되도록 조정 L2보다 안정적임
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')

# L2 정규화 : 각 행의 제곱의 합이 1이 되도록 조정
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 normalized data : \n", data_normalized_l1)
print("\nL1 normalized data : \n", data_normalized_l2)
