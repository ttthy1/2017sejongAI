# 라이브러리 불러오기
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 데이터 포인트를 생성하고 모델 학습시킴. 먼저 생성할 데이터 포인트 수를 정의
num_points = 1200

# 데이터 생성에 사용할 매개변수 정의. 예제에서는 y = mx + c 라는 선형 모델을 사용
data = []
m = 0.2
c = 0.5
for i in range(num_points):
    # x 생성하기
    x = np.random.normal(0.0, 0.8)

    # 노이즈 생성(데이터에 약간의 변화를 주기 위함)
    noise = np.random.normal(0.0, 0.04)

    # 등식의 y 값 계산
    y = m*x + c + noise

    data.append([x, y])

# 반복문이 끝나면, 데이터를 입력변수(x)와 출력변수(y)로 나누기
x_data = [d[0] for d in data]
y_data = [d[1] for d in data]

# 생성된 데이터를 그래프로 그리기
plt.plot(x_data, y_data, 'ro')
plt.title('Input data')
plt.show()

# 퍼셉트론에 대한 가중치(W)와 바이어스(b) 생성하기
# 가중치는 균등난수발생기로 생성하고, 바이어스는 ‘0’으로 지정
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# ‘y’에 대한 등식 정의하기
y = W * x_data + b

# 손실(loss) 계산방법 정의
loss = tf.reduce_mean(tf.square(y - y_data))

# Gradient descent 옵티마이저 정의하고, 손실함수 지정
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 모든 변수 초기화하기
init = tf.initialize_all_variables()

# 텐서플로우 세션 생성 및 실행
sess = tf.Session()
sess.run(init)

# 학습과정 시작
# 반복문 시작
num_iterations = 10
for step in range(num_iterations):
    # 세션 실행하기
    sess.run(train)

    # 진행상태를 화면에 출력하기
    print('\nITERATION', step+1)
    print('W =', sess.run(W)[0])
    print('b =', sess.run(b)[0])
    print('loss =', sess.run(loss))

    # 입렉 데이터를 그래프로 그리기
    plt.plot(x_data, y_data, 'ro')

    # 예측 결과에 대한 직선 그래프 그리기
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))

    # 그래프에 대한 매개변수 설정하기
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.title('Iteration ' + str(step+1) + ' of ' + str(num_iterations))
    plt.show()
