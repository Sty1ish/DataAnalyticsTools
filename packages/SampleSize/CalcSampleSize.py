import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# 샘플 사이즈계산 방법
# 참고해야할 논문 : https://vle.upm.edu.ph/pluginfile.php/223262/mod_resource/content/1/Power%20and%20Sample%20Size%20Determination%20reading.pdf
# 샘플 사이즈 계산 :https://probability.tools/calculators/sample-size


def _proportion_estimate_sample_size(p, e, z):
    return ((z**2) * p * (1 - p)) / (e**2)


def _mean_estimate_sample_size(sigma, e, z):
    return ((z * sigma) / e) ** 2


def _finite_population_correction(sample_size, population_size):
    return (sample_size * population_size) / (sample_size + population_size - 1)


def _draw_sample_size_curve_known_N(draw_type='proportion', **kwargs):
    # 데이터 파싱 및 연산
    if draw_type == 'proportion':
        # 변수 파싱
        # population_size이 있는 경우 보정이 들어가는 방식
        population_size = kwargs.get('population_size', None)
        p = kwargs.get('p')
        e = kwargs.get('e')
        confidence_level = kwargs.get('confidence_level')

        # 필수 파라미터 검증
        if p is None or e is None or confidence_level is None:
            raise ValueError(
                "'p', 'e', and 'confidence_level' are required for proportion type")

        # 플롯을 위한 값 연산
        z_value = norm.ppf((1 + confidence_level) / 2)
        default_sample_size = _proportion_estimate_sample_size(p, e, z_value)

        # 허용하는 에러 범위
        error_range = [i for i in np.arange((e * 0.1), (e * 3), (e * 0.01))]

        # 샘플 사이즈 범위 (허용하는 에러 범위에 따른 샘플 사이즈 변화)
        if population_size is not None:
            sample_range = [_finite_population_correction(_proportion_estimate_sample_size(
                p, error_rate, z_value), population_size) for error_rate in error_range]
        else:
            sample_range = [_proportion_estimate_sample_size(
                p, error_rate, z_value) for error_rate in error_range]

    elif draw_type == 'mean':
        # 변수 파싱
        # population_size이 있는 경우 보정이 들어가는 방식
        population_size = kwargs.get('population_size', None)
        sigma = kwargs.get('sigma')
        e = kwargs.get('e')
        confidence_level = kwargs.get('confidence_level')

        # 필수 파라미터 검증
        if sigma is None or e is None or confidence_level is None:
            raise ValueError(
                "'sigma', 'e', and 'confidence_level' are required for mean type")

        # 플롯을 위한 값 연산
        z_value = norm.ppf((1 + confidence_level) / 2)
        default_sample_size = _mean_estimate_sample_size(sigma, e, z_value)

        # 허용하는 에러 범위
        error_range = [i for i in np.arange((e * 0.1), (e * 3), (e * 0.01))]

        # 샘플 사이즈 범위 (허용하는 에러 범위에 따른 샘플 사이즈 변화)
        if population_size is not None:
            sample_range = [_finite_population_correction(_mean_estimate_sample_size(
                sigma, error_rate, z_value), population_size) for error_rate in error_range]
        else:
            sample_range = [_mean_estimate_sample_size(
                sigma, error_rate, z_value) for error_rate in error_range]

    else:
        raise ValueError("draw_type must be either 'proportion' or 'mean'")

    # 플롯 그리기
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # 제목 생성 (population_size가 None이면 "Infinite" 표시)
    pop_text = f'{population_size:,}' if population_size is not None else 'Infinite'
    if draw_type == 'proportion':
        title = f"""
        Required Sample Size by Error Rate 
        (Population: {pop_text}, Confidence: {confidence_level*100:.0f}%, p={p}, error={e})
        """
    else:
        title = f"""
        Required Sample Size by Error Rate
        (Population: {pop_text}, Confidence: {confidence_level*100:.0f}%, σ={sigma}, error={e})
        """

    ax1.set_title(title, fontsize=16)
    ax1.set_xlabel('Error Rate')
    ax1.set_ylabel('Sample Size')
    ax1.plot(error_range, sample_range, color='b')
    ax1.tick_params(axis='y')
    ax1.axvline(e, color='black', alpha=0.4, linestyle='--')
    ax1.text(e, max(sample_range)*0.98, f'{e:.3%}')
    ax1.axhline(default_sample_size, color='black', alpha=0.4, linestyle='--')
    ax1.text(e * 3 * 0.98, default_sample_size, f'{default_sample_size:,.0f}')

    plt.show()


def sample_size_proportion(p=0.5,
                           error=0.05,
                           confidence_level=0.95,
                           population_size=None,
                           draw_plot=True):
    """_summary_

    Args:
        p (float, optional): 어떤 타겟 변수가 발생할 확률 (모르는 경우 0.5를 사용). Defaults to 0.5.
        error (float, optional): 허용 오차 (Margin of Error). Defaults to 0.05.
        confidence_level (float, optional): 신뢰 수준. Defaults to 0.95.
        population_size (int, optional): 모집단 크기 (모르는 경우 주지 않아도 무방, 알고 있는 경우 Finite Population Correction 보정에 사용) Defaults to None.

    Returns:
        sample_size (float): 필요한 샘플 사이즈
    """
    z_value = norm.ppf((1 + confidence_level) / 2)
    sample_size = _proportion_estimate_sample_size(p, error, z_value)

    if population_size is not None:
        sample_size = _finite_population_correction(
            sample_size, population_size)

    if draw_plot:
        _draw_sample_size_curve_known_N(
            draw_type='proportion',
            p=p,
            e=error,
            confidence_level=confidence_level,
            population_size=population_size)
    return sample_size


def sample_size_mean(sigma,
                     error=0.05,
                     confidence_level=0.95,
                     population_size=None,
                     draw_plot=True):
    """_summary_

    Args:
        sigma (float): 모집단의 표준편차 (표본의 표준편차로 대체 가능, 반드시 필요 함 -> 사전 조사를 통해서라도 얻을 것)
        error (float, optional): 허용 오차 (Margin of Error). Defaults to 0.05.
        confidence_level (float, optional): 신뢰 수준. Defaults to 0.95.
        population_size (int, optional): 모집단 크기 (모르는 경우 주지 않아도 무방, 알고 있는 경우 Finite Population Correction 보정에 사용) Defaults to None.

    Returns:
        sample_size (float): 필요한 샘플 사이즈
    """
    z_value = norm.ppf((1 + confidence_level) / 2)
    sample_size = _mean_estimate_sample_size(sigma, error, z_value)

    if population_size is not None:
        sample_size = _finite_population_correction(
            sample_size, population_size)

    if draw_plot:
        _draw_sample_size_curve_known_N(
            draw_type='mean',
            sigma=sigma,
            e=error,
            confidence_level=confidence_level,
            population_size=population_size)

    return sample_size
