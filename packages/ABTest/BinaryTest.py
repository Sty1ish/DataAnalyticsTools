import pandas as pd
import polars as pl
from scipy.stats import fisher_exact
from statsmodels.stats import proportion
from statsmodels.stats.contingency_tables import Table2x2

# Visualize
import matplotlib.pyplot as plt


def fisher_exact_test(xa, xb, na, nb, alpha, alternative='two-sided'):
    """
    신뢰구간을 구하는 공식 : https://rpubs.com/mbh038/990902
    검정 통계량 = Odd Ratio
    신뢰 구간은 윌슨 점수를 이용해서 계산 (https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval) -> 소규모 표본에도 적용 가능

    Args:
        xa (int): 그룹 A의 성공 횟수
        xb (int): 그룹 B의 성공 횟수
        na (int): 그룹 A의 표본 크기
        nb (int): 그룹 B의 표본 크기
        alpha (float): 유의 수준
        alternative (str, optional): 대립 가설의 방향 ('two-sided', 'smaller', 'larger'). Defaults to 'two-sided'.

    Returns:
        dict: 검정통계량 종류, odds_ratio, p-value, A 집단 신뢰구간, B 집단 신뢰구간, 두 집단 차이의 신뢰구간
    """
    table = [[xa, na - xa],
             [xb, nb - xb]]

    # alternative 매개변수는 scipy의 fisher_exact 함수에서 사용되는 방식과 일치하도록 변환
    if alternative == 'smaller':
        alternative = 'less'
    elif alternative == 'larger':
        alternative = 'greater'

    # Fisher p-value
    statistics, p_value = fisher_exact(table, alternative=alternative)

    # Exact OR + CI
    t = Table2x2(table)
    oddsratio = t.oddsratio
    diff_lower, diff_upper = t.oddsratio_confint()

    # 이항 비율 신뢰구간 계산 (Wilson score interval)
    a_lower, a_upper = proportion.proportion_confint(
        xa, na, alpha=alpha, method='wilson')
    b_lower, b_upper = proportion.proportion_confint(
        xb, nb, alpha=alpha, method='wilson')

    return {'statistic_type': 'odds_ratio',
            'statistic': oddsratio,
            'p_value': p_value,
            'ci_a': (a_lower, a_upper),
            'ci_b': (b_lower, b_upper),
            'ci_diff': (diff_lower, diff_upper)}


def z_test_proportions(xa, xb, na, nb, alpha, alternative='two-sided'):
    """
    이표본 비율 Z Test의 검정 결과를 반환하는 함수

    Args:
        xa (int): 그룹 A의 성공 횟수
        xb (int): 그룹 B의 성공 횟수
        na (int): 그룹 A의 표본 크기
        nb (int): 그룹 B의 표본 크기
        alpha (float): 유의 수준
        alternative (str): 대립 가설의 방향 ('two-sided', 'smaller', 'larger')

    Returns:
        dict: 검정통계량 종류, z, p-value, A 집단 신뢰구간, B 집단 신뢰구간, 두 집단 차이의 신뢰구간
    """
    z, p_value = proportion.proportions_ztest(
        count=[xa, xb],
        nobs=[na, nb],
        alternative=alternative  # two-sided, smaller, larger
    )

    # 이항 비율 신뢰구간 계산 (Wald score interval)
    (a_lower, b_lower), (a_upper, b_upper) = proportion.proportion_confint(
        [xa, xb],
        nobs=[na, nb],
        alpha=alpha,
        method='normal'
    )

    diff_lower, diff_upper = proportion.confint_proportions_2indep(
        count1=xa,
        nobs1=na,
        count2=xb,
        nobs2=nb,
        method='wald',
        alpha=alpha
    )

    return {'statistic_type': 'z',
            'statistic': z,
            'p_value': p_value,
            'ci_a': (a_lower, a_upper),
            'ci_b': (b_lower, b_upper),
            'ci_diff': (diff_lower, diff_upper)}


def is_NormalApproximation(na, nb, pa, pb):
    """
    이항분포의 정규성 근사 조건 확인 : np, npq가 10이상인 경우 CLT에 의해 정규성 근사가 가능하다고 판단
    5가 아닌 10을 대상으로 한 근거 : https://en.wikipedia.org/wiki/Two-proportion_Z-test
    Args:
        na (int): 그룹 A의 표본 크기
        nb (int): 그룹 B의 표본 크기
        pa (float): 그룹 A의 성공 확률
        pb (float): 그룹 B의 성공 확률

    Returns:
        bool: 정규성 근사 가능 여부 (True: 가능, False: 불가능)
    """

    if (na * pa < 10) or \
       (na * (1 - pa) < 10) or \
       (nb * pb < 10) or \
       (nb * (1 - pb) < 10):
        return False
    elif na + nb < 20:
        return False
    else:
        return True


def two_group_proportion_test_from_data(data,
                                        group_col='group',
                                        target_col='is_purchase',
                                        alternative='two-sided',
                                        alpha=0.05):
    """
    두 집단 비율 차이에 대한 Z-검정 (정규근사를 기반으로 하고 있어, 정규성 근사 조건이 만족되어야 함)
    기본적으로는 양측 검정을 전제에 두고 있음 (alternative='two-sided'), 단측 검정도 가능 (alternative='smaller' or 'larger')
    https://en.wikipedia.org/wiki/Two-proportion_Z-test

    Args:
        data (pl.DataFrame): 입력 데이터 프레임
        group_col (str, optional): 그룹을 나타내는 열 이름. Defaults to 'group'.
        target_col (str, optional): 타겟 변수를 나타내는 열 이름. Defaults to 'is_purchase'.
        alternative (str, optional): 대립 가설의 방향 ('two-sided', 'smaller', 'larger'). Defaults to 'two-sided'.
        alpha (float, optional): 유의 수준. Defaults to 0.05.

    Returns:
        dict: 검정 통계량(z), p-value, 그룹 레이블, 그룹별 비율(pa, pb), 그룹별 신뢰구간(ci_a, ci_b)
    """

    # 입력 데이터가 pandas DataFrame인 경우, polars DataFrame으로 변환
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    # 테스트 변수 추출
    agg_df = (
        data
        .group_by(group_col)
        .agg(
            pl.count(target_col).alias("n"),
            pl.sum(target_col).alias("x"),
            (pl.sum(target_col) / pl.count(target_col)).alias("p")
        )
        .sort(group_col)   # 순서 고정
    )

    # agg_df에서 두개 이상의 그룹이 나온경우, 두 집단의 검정이 아니기 때문에 에러 발생
    if agg_df.height != 2:
        raise ValueError(f"Expected exactly 2 groups, got {agg_df.height}")

    (groupA_label, na, xa, pa), (groupB_label, nb, xb, pb) = agg_df.rows()

    # 정규성 근사 조건 확인
    if is_NormalApproximation(na, nb, pa, pb):
        test_statistics = z_test_proportions(
            xa, xb, na, nb, alpha, alternative)
    else:
        test_statistics = fisher_exact_test(xa, xb, na, nb, alpha, alternative)

    # 검정 방식에 따른 통계량 값 분할
    z = test_statistics['statistic']
    p_value = test_statistics['p_value']
    (a_lower, a_upper) = test_statistics['ci_a']
    (b_lower, b_upper) = test_statistics['ci_b']
    (diff_lower, diff_upper) = test_statistics['ci_diff']

    # 검정 결과 출력
    if alternative == "two-sided":
        H0 = f"p({groupA_label}) = p({groupB_label})"
        H1 = f"p({groupA_label}) ≠ p({groupB_label})"

    elif alternative == "smaller":
        H0 = f"p({groupA_label}) ≥ p({groupB_label})"
        H1 = f"p({groupA_label}) < p({groupB_label})"

    elif alternative == "larger":
        H0 = f"p({groupA_label}) ≤ p({groupB_label})"
        H1 = f"p({groupA_label}) > p({groupB_label})"

    if test_statistics['statistic_type'] == 'z':
        HEADER = "Two-Proportion Z-Test가 수행되었습니다"
        CALC_INTERVAL_METHOD = "Wald score interval"

    elif test_statistics['statistic_type'] == 'odds_ratio':
        HEADER = "Fisher's Exact Test가 수행되었습니다 (정규성 근사 조건 미 충족)"
        CALC_INTERVAL_METHOD = "Wilson score interval"

    else:
        HEADER = "알 수 없는 검정 방식이 수행되었습니다"
        CALC_INTERVAL_METHOD = "알 수 없음"

    print(f"""
          {HEADER}
            ------------------------------------------------
            이표본 {'양측' if alternative == 'two-sided' else '단측'} 비율 검정 결과
            ------------------------------------------------
            유의수준 α = {alpha}   신뢰수준 = {(1-alpha)*100:.1f}%

            그룹 A: {groupA_label} (n={na}, x={xa}, p={pa:.4f} {str((1-alpha) * 100)}% CI: {a_lower:.4f} ~ {a_upper:.4f})
            그룹 B: {groupB_label} (n={nb}, x={xb}, p={pb:.4f} {str((1-alpha) * 100)}% CI: {b_lower:.4f} ~ {b_upper:.4f})
            * 신뢰구간은 {CALC_INTERVAL_METHOD} 방법으로 계산 되었습니다
            ------------------------------------------------
            H0: {H0}
            H1: {H1}
            diff : {pa - pb:.4f} ({str((1-alpha) * 100)}% CI: {diff_lower:.4f} ~ {diff_upper:.4f}) | p-value = {p_value:.5f}
            결론: {"귀무가설 기각" if p_value < alpha else "귀무가설 기각 실패"}
            ------------------------------------------------
          """)

    # visualize
    # visualize
    plt.figure(figsize=(10, 3))
    plt.ylim(-0.5, 1.5)
    # A 그룹
    plt.errorbar(x=pa*100,
                 y=0,
                 xerr=[[(pa - a_lower) * 100], [(a_upper - pa) * 100]],
                 fmt='o--',
                 capsize=5,
                 color='blue'
                 )
    # B 그룹
    plt.errorbar(x=pb * 100,
                 y=1,
                 xerr=[[(pb - b_lower) * 100], [(b_upper - pb) * 100]],
                 fmt='o--',
                 capsize=5,
                 color='red'
                 )
    plt.yticks(range(2), [groupA_label, groupB_label])
    plt.title(f'Conversion Rate ({str((1-alpha) * 100)}% Confidence Level)')
    plt.xlabel('Conversion Rate (%)')
    plt.ylabel('Group')
    plt.show()

    return {
        "z": z,
        "p_value": p_value,
        "label_a": groupA_label,
        "pa": pa,
        "label_b": groupB_label,
        "pb": pb,
        "ci_a": (a_lower, a_upper),
        "ci_b": (b_lower, b_upper),
    }


def two_group_proportion_test_from_summary(xa, xb,
                                           na, nb,
                                           alpha=0.05,
                                           alternative='two-sided'):
    groupA_label = "A"
    groupB_label = "B"
    pa = xa / na
    pb = xb / nb

    if is_NormalApproximation(na, nb, pa, pb):
        test_statistics = z_test_proportions(
            xa, xb, na, nb, alpha, alternative)
    else:
        test_statistics = fisher_exact_test(xa, xb, na, nb, alpha, alternative)
        # 검정 방식에 따른 통계량 값 분할
    z = test_statistics['statistic']
    p_value = test_statistics['p_value']
    (a_lower, a_upper) = test_statistics['ci_a']
    (b_lower, b_upper) = test_statistics['ci_b']
    (diff_lower, diff_upper) = test_statistics['ci_diff']

    # 검정 결과 출력
    if alternative == "two-sided":
        H0 = f"p({groupA_label}) = p({groupB_label})"
        H1 = f"p({groupA_label}) ≠ p({groupB_label})"

    elif alternative == "smaller":
        H0 = f"p({groupA_label}) ≥ p({groupB_label})"
        H1 = f"p({groupA_label}) < p({groupB_label})"

    elif alternative == "larger":
        H0 = f"p({groupA_label}) ≤ p({groupB_label})"
        H1 = f"p({groupA_label}) > p({groupB_label})"

    if test_statistics['statistic_type'] == 'z':
        HEADER = "Two-Proportion Z-Test가 수행되었습니다"
        CALC_INTERVAL_METHOD = "Wald score interval"

    elif test_statistics['statistic_type'] == 'odds_ratio':
        HEADER = "Fisher's Exact Test가 수행되었습니다 (정규성 근사 조건 미 충족)"
        CALC_INTERVAL_METHOD = "Wilson score interval"

    else:
        HEADER = "알 수 없는 검정 방식이 수행되었습니다"
        CALC_INTERVAL_METHOD = "알 수 없음"

    print(f"""
          {HEADER}
            ------------------------------------------------
            이표본 {'양측' if alternative == 'two-sided' else '단측'} 비율 검정 결과
            ------------------------------------------------
            유의수준 α = {alpha}   신뢰수준 = {(1-alpha)*100:.1f}%

            그룹 A: {groupA_label} (n={na}, x={xa}, p={pa:.4f} {str((1-alpha) * 100)}% CI: {a_lower:.4f} ~ {a_upper:.4f})
            그룹 B: {groupB_label} (n={nb}, x={xb}, p={pb:.4f} {str((1-alpha) * 100)}% CI: {b_lower:.4f} ~ {b_upper:.4f})
            * 신뢰구간은 {CALC_INTERVAL_METHOD} 방법으로 계산 되었습니다
            ------------------------------------------------
            H0: {H0}
            H1: {H1}
            diff : {pa - pb:.4f} ({str((1-alpha) * 100)}% CI: {diff_lower:.4f} ~ {diff_upper:.4f}) | p-value = {p_value:.5f}
            결론: {"귀무가설 기각" if p_value < alpha else "귀무가설 기각 실패"}
            ------------------------------------------------
          """)

    # visualize
    # visualize
    plt.figure(figsize=(10, 3))
    plt.ylim(-0.5, 1.5)
    # A 그룹
    plt.errorbar(x=pa*100,
                 y=0,
                 xerr=[[(pa - a_lower) * 100], [(a_upper - pa) * 100]],
                 fmt='o--',
                 capsize=5,
                 color='blue'
                 )
    # B 그룹
    plt.errorbar(x=pb * 100,
                 y=1,
                 xerr=[[(pb - b_lower) * 100], [(b_upper - pb) * 100]],
                 fmt='o--',
                 capsize=5,
                 color='red'
                 )
    plt.yticks(range(2), [groupA_label, groupB_label])
    plt.title(f'Conversion Rate ({str((1-alpha) * 100)}% Confidence Level)')
    plt.xlabel('Conversion Rate (%)')
    plt.ylabel('Group')
    plt.show()

    return {
        "z": z,
        "p_value": p_value,
        "label_a": groupA_label,
        "pa": pa,
        "label_b": groupB_label,
        "pb": pb,
        "ci_a": (a_lower, a_upper),
        "ci_b": (b_lower, b_upper),
    }
