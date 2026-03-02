import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import polars as pl
import pandas as pd
import numpy as np


def two_group_numerical_test_from_data(data,
                                       group_col='group',
                                       target_col='is_purchase',
                                       alternative='two-sided',
                                       alpha=0.05):
    """
    두 집단의 수치형 차이에 대한 Welch's T-Test (이 분포는 정규 / 등분산을 가정하지 않음) 검정을 수행하는 함수입니다.
    거의 모든 데이터에 대한 적용이 가능합니다 (단 매우 작은 표본, ordinal 데이터, 비모수적 데이터는 권장하지 않음).
    기본적으로는 양측 검정을 전제에 두고 있음 (alternative='two-sided'), 단측 검정도 가능 (alternative='two-sided', 'less', 'greater')
    https://en.wikipedia.org/wiki/Two-proportion_Z-test

    Args:
        data (pl.DataFrame): 입력 데이터 프레임
        group_col (str, optional): 그룹을 나타내는 열 이름. Defaults to 'group'.
        target_col (str, optional): 타겟 변수를 나타내는 열 이름. Defaults to 'is_purchase'.
        alternative (str, optional): 대립 가설의 방향 ('two-sided', 'less', 'greater'). Defaults to 'two-sided'.
        alpha (float, optional): 유의 수준. Defaults to 0.05.

    Returns:
        dict: 검정 통계량(z), p-value, 그룹 레이블, 그룹별 비율(pa, pb), 그룹별 신뢰구간(ci_a, ci_b)
    """

    # pandas -> polars
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    # 그룹별 데이터 추출
    agg_df = (
        data
        .group_by(group_col)
        .agg(
            pl.count(target_col).alias("n"),
            pl.mean(target_col).alias("mean"),
            pl.var(target_col).alias("var")
        )
        .sort(group_col)   # 순서 고정
    )

    # 이표본 T-Test인지 확인
    if agg_df.shape[0] != 2:
        raise ValueError("두개의 group 만 존재해야 합니다.")

    # 연산에 필요한 데이터 추출
    (g1, n1, m1, v1), (g2, n2, m2, v2) = agg_df.rows()
    group1 = data.filter(pl.col(group_col) == g1)[target_col]
    group2 = data.filter(pl.col(group_col) == g2)[target_col]

    # 표본의 크기가 30 이하인 경우, Mann-Whitney U Test 권장
    if (group1.shape[0] <= 30) and (group2.shape[0] <= 30):
        raise ValueError("매우 작은 숫자의 표본으로 Mann-Whitney U Test를 권장합니다")

    # ttest_ind에서 equal_var=False로 하면 Welch's t-test로 동작
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    result = stats.ttest_ind(
        group1,
        group2,
        equal_var=False,
        alternative=alternative
    )

    statistics = result.statistic
    p_value = result.pvalue
    df = result.df
    (diff_lower, diff_upper) = result.confidence_interval(confidence_level=1-alpha)

    # CLT가 성립함을 가정했으므로(raise Error를 통해 각 그룹의 n > 30임을 전제), 신뢰구간 계산은 T분포의 신뢰구간 계산 공식으로 계산
    g1_lower, g1_upper = stats.t.interval(
        confidence=1-alpha,
        df=n1-1,
        loc=m1,
        scale=np.sqrt(v1)/np.sqrt(n1)
    )

    g2_lower, g2_upper = stats.t.interval(
        confidence=1-alpha,
        df=n2-1,
        loc=m2,
        scale=np.sqrt(v2)/np.sqrt(n2)
    )

    # 가설 설정
    if alternative == "two-sided":
        H0 = f"μ({g1}) = μ({g2})"
        H1 = f"μ({g1}) ≠ μ({g2})"
    elif alternative == "smaller":
        H0 = f"μ({g1}) ≥ μ({g2})"
        H1 = f"μ({g1}) < μ({g2})"
    else:
        H0 = f"μ({g1}) ≤ μ({g2})"
        H1 = f"μ({g1}) > μ({g2})"

    print(f"""
        ------------------------------------------------
        이표본 {'양측' if alternative == 'two-sided' else '단측'} Welch's T-Test 결과
        ------------------------------------------------
        유의수준 α = {alpha}   신뢰수준 = {(1-alpha)*100:.1f}%

        그룹 A: {g1} (n={n1}, mean={m1:.4f}, var={v1:.4f} {str((1-alpha) * 100)}% CI: {g1_lower:.4f} ~ {g1_upper:.4f})
        그룹 B: {g2} (n={n2}, mean={m2:.4f}, var={v2:.4f} {str((1-alpha) * 100)}% CI: {g2_lower:.4f} ~ {g2_upper:.4f})
        * 신뢰구간은 T-분포 근사를 통해 계산되었습니다 (두 그룹의 표본수가 30보다 커서 CLT 가정)
        ------------------------------------------------
        H0: {H0}
        H1: {H1}
        diff : {m1 - m2:.4f} ({str((1-alpha) * 100)}% CI: {diff_lower:.4f} ~ {diff_upper:.4f}) | p-value = {p_value:.5f}
        결론: {"귀무가설 기각" if p_value < alpha else "귀무가설 기각 실패"}
        ------------------------------------------------
        """)

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    ax = sns.kdeplot(x=target_col, data=data, hue=group_col, fill=True, ax=ax)
    ymin, ymax = ax.get_ylim()
    ax.set_title(
        'Distribution and Confidence Intervals of Group Means', fontsize=16)
    ax.hlines(y=(ymin + (ymax - ymin) * 0.1), xmin=group1.mean(),
              xmax=group2.mean(), color='black', linewidth=2)
    ax.axvline(group1.mean(), linestyle=':', color='black')
    ax.axvline(group2.mean(), linestyle=':', color='red')
    ax.text(group1.mean(), 0, f'{group1.mean():.4f}')
    ax.text(group2.mean(), 0, f'{group2.mean():.4f}')
    ax.text((group2.mean()+group1.mean())/2, (ymin + (ymax - ymin) * 0.11),
            f'{abs(group2.mean()-group1.mean()):.4f}')
    ax.text(x=group1.mean(), y=0, s=round(
        group1.mean(), 1), color='black')
    ax.text(x=group2.mean(), y=0, s=round(
        group2.mean(), 1), color='red')
    ax.axvspan(g1_lower, g1_upper, alpha=0.2, color='black')
    ax.axvspan(g2_lower, g2_upper, alpha=0.2, color='red')

    return {
        "statistics": statistics,
        "p_value": p_value,
        "label_a": g1,
        "mean_a": m1,
        "label_b": g2,
        "mean_b": m2,
        "ci_a": (g1_lower, g1_upper),
        "ci_b": (g2_lower, g2_upper),
        "diff": m1 - m2,
        "diff_ci": (diff_lower, diff_upper)
    }
