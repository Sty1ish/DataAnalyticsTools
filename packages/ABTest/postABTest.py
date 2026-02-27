# AB 테스트 전 실험 설계 단계에서 필요한 함수들을 고민합니다
# MDE 계산 / 샘플 사이즈 계산 / 검정력 계산 등등
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats import proportion


def binary_sample_size_calc(diff,
                            prop2,
                            power,
                            ratio,
                            alpha=0.05,
                            value=0,
                            alternative='two-sided',
                            plot_curve=True):
    """
    참고 수식 : https://statistics.tools/sample-size-proportions-calculator
    - 성공 비율이 50%인 경우 분산이 최대가 되므로, 필요한 샘플수가 증가합니다 (p를 모르는 경우 50%로 가정하는 이유)
    - 기대하는 효과 사이즈가 작은 경우, 필요한 샘플수가 증가합니다 (효과 사이즈가 작을수록, 차이를 검출하기 어려워지므로)
    - 유의수준이 높을수록 더 많은 샘플사이즈가 필요합니다 (더 높은 신뢰도를 위해서는 더 많은 샘플이 필요)

    Args:
        diff (float) : (MDE) p1 - p2, 기대하는 효과 사이즈 (예시: 0.02는 2%p의 차이를 기대한다는 의미)
        prop2 (float) : 대조군(p2)의 성공 확률 (예상되는 성공 확률, p1 = p2 + diff)
        power (float) : 대립가설이 참일때 이를 사실로 결정할 확률
        ratio (float) : 샘플간 크기 비율 (nobs2 = ratio * nobs1) -> 예시: ratio=1은 대조군과 실험군의 샘플수가 동일, ratio=2는 실험군이 대조군보다 2배 많음
        alpha (float) : 유의수준, 제1종 오류 확률 (예시: 0.05는 5%의 확률로 귀무가설이 참인데도 이를 기각하는 오류를 허용한다는 의미)
        value (float) : Currently only value=0, i.e. equality testing, is supported
        alternative (str) : ‘two-sided’ (default), ‘larger’, ‘smaller’ 지원, 기본적으로 양측 검정으로 사용할 것

    Returns:
        nobs1 (float) : 그룹 1에 필요한 샘플 사이즈
        nobs2 (float) : 그룹 2에 필요한 샘플 사이즈
    """

    # 두 그룹의 표본 숫자 계산 결과 (함수 사용)
    n1 = proportion.samplesize_proportions_2indep_onetail(
        diff=diff,
        prop2=prop2,
        power=power,
        ratio=ratio,
        alpha=alpha,
        alternative=alternative
    )

    # Return용 값은 n1, n2의 형태로 변환 후 반환
    n1, n2 = n1, n1 * ratio

    # 현재 조건에서, 표본 크기만 변화할 때 검정력의 변화
    if plot_curve:
        # n1의 2배까지 10단위로 샘플 사이즈 범위 설정
        sample_size_list = np.arange(10, int(max(n1, n2) * 4), 10)
        sample_target_power = []

        for idx, n in enumerate(sample_size_list):
            target_power = proportion.power_proportions_2indep(diff=diff,
                                                               prop2=prop2,
                                                               nobs1=n,
                                                               ratio=ratio,
                                                               alpha=alpha,
                                                               alternative=alternative
                                                               )
            sample_target_power.append(target_power.power)

            if target_power.power >= 0.99:
                sample_size_list = sample_size_list[:idx+1]
                break

        plt.figure(figsize=(8, 5))
        plt.plot(sample_size_list, sample_target_power, linestyle='-')
        plt.axhline(power, color='r', linestyle='--',
                    label=f'Target power = {power:.2f}')
        plt.title(f"Power vs Sample Size (n1 = {n1:.0f}, n2 = {n2:.0f})")
        plt.xlabel(f"Sample size per group (n1)")
        plt.ylabel("Power")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.show()

    return {'group1': n1, 'group2': n2}
