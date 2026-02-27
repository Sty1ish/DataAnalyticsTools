import matplotlib.pyplot as plt
import numpy as np

# 샘플 사이즈계산
# 참고해야할 논문 : https://vle.upm.edu.ph/pluginfile.php/223262/mod_resource/content/1/Power%20and%20Sample%20Size%20Determination%20reading.pdf
# 샘플 사이즈 계산 :https://probability.tools/calculators/sample-size


# 지금 아래 수식은, 비율 데이터에 대한 샘플 사이즈 계산 수식입니다

def sample_size(N, z, p, e):
    top = ((z**2) * p * (1 - p)) / (e ** 2)
    bottom = 1 + (((z**2) * p * (1 - p)) / ((e ** 2) * N))
    return top / bottom


N = 1723590615    # 모수(전체 사이즈)
z = 1.96          # 신뢰수준 Normal 95% = 1.96
p = 0.5           # 베르누이 분포 가정 (True - False 비율 5:5)
e = 0.001         # 오차 한계 (%)

# Calc
sample_size(N, z, p, e)


# GRAPH
e_range = [i for i in np.arange((e * 0.1), (e * 3), (e * 0.01))]
sample_range = [sample_size(N, z, p, e_r) for e_r in e_range]
sample_size_range = [round(e_r, 6) * N for e_r in e_range]

plt.plot(e_range, sample_size_range, marker='s', color='b')
plt.xlabel('Max Error')
plt.ylabel('Sample(Error)_size')

fig, ax1 = plt.subplots(figsize=(8, 6))
color_1 = 'tab:blue'
ax1.set_title(
    '(Z = 1.96) Error Rate via sample size & Error size', fontsize=16)
ax1.set_xlabel('Error Rate')
ax1.set_ylabel('Sample Size', fontsize=14, color=color_1)
ax1.plot(e_range, sample_range, color='b')
ax1.tick_params(axis='y', labelcolor=color_1)

# right side with different scale
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color_2 = 'tab:red'
ax2.set_ylabel('Error Size (Statistical)', fontsize=14, color=color_2)
ax2.plot(e_range, sample_size_range, color='r')
ax2.tick_params(axis='y', labelcolor=color_2)


ax2.axvline(0.00036, color='black', alpha=0.4)
ax1.text(0.00036, 0, '0.036%')
ax1.axhline(8000000, color='black', alpha=0.4)
ax1.text(0, 8000000, '8,000,000')

# 실제 선정한 위치
ax2.axvline(0.001, color='green', alpha=0.4)
ax1.text(0.001, 0, '0.1%')
