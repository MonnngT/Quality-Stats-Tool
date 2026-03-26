import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import re

st.set_page_config(page_title="全能版质量统计中心", layout="wide")
st.title("📊 Web 版 Minitab - 全能质量统计工具")

# 侧边栏菜单
analysis_type = st.sidebar.radio(
    "请选择分析工具：",
    (
        "0. 正态性检验 (Normality Test)",
        "1. 单样本 T检验 (1-Sample t)",
        "2. 双样本 T检验 (2-Sample t)",
        "3. 配对 T检验 (Paired t)",
        "4. 单比例检验 (1-Proportion)",
        "5. 双比例检验 (2-Proportion)",
        "6. 卡方检验 (Chi-Square)",
        "7. 单因素方差分析 (One-Way ANOVA)",
        "8. 双因素方差分析 (Two-Way ANOVA)"
    )
)

def parse_data(input_string, dtype=float):
    if not input_string.strip():
        return []
    raw_data = re.split(r'[\s,]+', input_string.strip())
    try:
        return [dtype(x) for x in raw_data if x]
    except ValueError:
        st.error("数据解析错误！请确保输入格式正确。")
        return []

# ================= 0. 正态性检验 =================
if analysis_type == "0. 正态性检验 (Normality Test)":
    st.header("正态性检验 (Shapiro-Wilk Test)")
    st.markdown("""
    > 🎯 **前提要求**：T检验、ANOVA 等连续型数据分析工具，均要求数据服从正态分布（钟形曲线）。
    > 📌 **判断标准**：**P > 0.05** 代表数据为正态分布（良民），可以放心使用后续高级工具；P < 0.05 代表数据异常或偏态。
    """)
    
    data_input = st.text_area("粘贴测试数据:", "10.1 9.8 10.5 10.0 10.2 9.9 10.3 10.1 9.7 10.4")
    
    if st.button("运行正态体检"):
        data = parse_data(data_input)
        if len(data) < 3:
            st.warning("至少需要3个数据点。")
        else:
            stat, p_value = stats.shapiro(data)
            st.subheader("💡 检验结果")
            st.write(f"- **P 值**: {p_value:.4f}")
            if p_value > 0.05:
                st.success("✅ P > 0.05，接受原假设：数据**服从**正态分布，可以放心使用后续高级统计工具。")
            else:
                st.error("❌ P < 0.05，拒绝原假设：数据**不服从**正态分布，请排查异常值或考虑数据转换。")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(data, kde=True, ax=ax1)
            ax1.set_title("Histogram")
            stats.probplot(data, dist="norm", plot=ax2)
            ax2.set_title("Normal Q-Q Plot")
            st.pyplot(fig)

# ================= 1. 单样本 T检验 =================
elif analysis_type == "1. 单样本 T检验 (1-Sample t)":
    st.header("单样本 T检验")
    st.markdown("""
    > 🎯 **应用场景**：比较【当前的一批产品均值】与【固定的设计目标值】是否有显著偏差。例如：抽测20台直流电机，验证平均转速是否偏离了3000 RPM的设计规格。
    > 📌 **判断标准**：**P < 0.05** 代表现状均值与目标值存在统计学上的**显著差异**；P >= 0.05 代表偏差属于正常波动范围。
    """)
    
    target_mean = st.number_input("输入目标均值:", value=3000.0)
    data_input = st.text_area("粘贴测试数据:", "2985 2990 3005 2980 2995")
    
    if st.button("运行分析"):
        data = parse_data(data_input)
        if len(data) >= 2:
            t_stat, p_value = stats.ttest_1samp(data, target_mean)
            st.write(f"- **样本均值**: {np.mean(data):.2f} | **P 值**: {p_value:.4f}")
            if p_value < 0.05: st.error("显著偏离目标值。")
            else: st.success("属正常波动，未显著偏离。")

# ================= 2. 双样本 T检验 =================
elif analysis_type == "2. 双样本 T检验 (2-Sample t)":
    st.header("双样本 T检验 (独立样本)")
    st.markdown("""
    > 🎯 **应用场景**：比较【两批相互独立的产品】的均值是否有显著差异。例如：对比A机床与B机床生产的零件尺寸差异，或对比验证“新工艺”的拉拔力是否真的比“旧工艺”高。
    > 📌 **判断标准**：**P < 0.05** 代表两组数据存在**显著差异**；P >= 0.05 代表两组处于同一水平。
    """)
    
    col1, col2 = st.columns(2)
    with col1: d1_in = st.text_area("组别 A (如旧工艺):", "15.2 14.8 15.5 14.9 15.1")
    with col2: d2_in = st.text_area("组别 B (如新工艺):", "16.1 15.9 16.5 16.2 16.0")
    
    if st.button("运行分析"):
        d1, d2 = parse_data(d1_in), parse_data(d2_in)
        if len(d1) >= 2 and len(d2) >= 2:
            _, p_lev = stats.levene(d1, d2)
            t_stat, p_value = stats.ttest_ind(d1, d2, equal_var=(p_lev > 0.05))
            st.write(f"- **方差齐性 P值**: {p_lev:.4f} | **T检验 P值**: {p_value:.4f}")
            if p_value < 0.05: st.error("两组数据存在显著差异！")
            else: st.success("两组数据无显著差异。")

# ================= 3. 配对 T检验 =================
elif analysis_type == "3. 配对 T检验 (Paired t)":
    st.header("配对 T检验 (Paired t-test)")
    st.markdown("""
    > 🎯 **应用场景**：比较【同一批对象】在受到某种处理【前】与【后】的变化。两组数据必须是一一对应的。例如：测量同一批定子在绝缘浸漆“前”和“后”的重量变化，以验证浸漆附着量。
    > 📌 **判断标准**：**P < 0.05** 代表处理前后发生了**显著变化**。
    """)
    col1, col2 = st.columns(2)
    with col1: d1_in = st.text_area("处理前 (Before):", "50 52 49 55 51")
    with col2: d2_in = st.text_area("处理后 (After):", "53 54 53 58 54")
    
    if st.button("运行分析"):
        d1, d2 = parse_data(d1_in), parse_data(d2_in)
        if len(d1) != len(d2) or len(d1) < 2:
            st.warning("两组数据数量必须相等且至少2个。")
        else:
            t_stat, p_value = stats.ttest_rel(d1, d2)
            st.write(f"- **差异均值**: {np.mean(np.array(d2)-np.array(d1)):.2f} | **P 值**: {p_value:.4f}")
            if p_value < 0.05: st.error("处理前后存在显著差异！")
            else: st.success("处理前后无显著差异。")

# ================= 4. 单比例检验 =================
elif analysis_type == "4. 单比例检验 (1-Proportion)":
    st.header("单比例检验")
    st.markdown("""
    > 🎯 **应用场景**：比较【当前的实际不良率】与【历史设定的目标不良率】是否有显著差异。例如：判断近期3%的外壳表面划伤率，是否已经显著高于了长期允许的2%上限目标。
    > 📌 **判断标准**：**P < 0.05** 代表实际不良率与目标值存在**显著偏差**。
    """)
    target_p = st.number_input("目标不良率/比率 (0~1):", value=0.02)
    col1, col2 = st.columns(2)
    with col1: count = st.number_input("发现的不良数:", value=15, step=1)
    with col2: nobs = st.number_input("抽检总数:", value=500, step=1)
    
    if st.button("运行分析"):
        stat, p_value = proportions_ztest(count, nobs, value=target_p)
        st.write(f"- **实际不良率**: {count/nobs:.4f} | **P 值**: {p_value:.4f}")
        if p_value < 0.05: st.error("实际比率与目标比率有显著差异！")
        else: st.success("实际比率与目标比率无显著差异。")

# ================= 5. 双比例检验 =================
elif analysis_type == "5. 双比例检验 (2-Proportion)":
    st.header("双比例检验")
    st.markdown("""
    > 🎯 **应用场景**：比较【两个不同群体】的不良率是否有显著差异。例如：对比供应商A和供应商B交货批次的不合格品率，评判哪家供应商的质量控制更稳定。
    > 📌 **判断标准**：**P < 0.05** 代表两组的不良率存在**显著差异**。
    """)
    col1, col2 = st.columns(2)
    with col1:
        c1 = st.number_input("组A 不良数:", value=12, step=1)
        n1 = st.number_input("组A 抽样数:", value=1000, step=1)
    with col2:
        c2 = st.number_input("组B 不良数:", value=5, step=1)
        n2 = st.number_input("组B 抽样数:", value=800, step=1)
        
    if st.button("运行分析"):
        counts = np.array([c1, c2])
        nobs = np.array([n1, n2])
        stat, p_value = proportions_ztest(counts, nobs)
        st.write(f"- **组A不良率**: {c1/n1:.4f} | **组B不良率**: {c2/n2:.4f} | **P 值**: {p_value:.4f}")
        if p_value < 0.05: st.error("两组不良率存在显著差异！")
        else: st.success("两组不良率无显著差异。")

# ================= 6. 卡方检验 =================
elif analysis_type == "6. 卡方检验 (Chi-Square)":
    st.header("卡方检验 (独立性检验)")
    st.markdown("""
    > 🎯 **应用场景**：判断两个【分类事件】之间是否相互关联。例如：分析“缺陷类型（缺料、缩孔、毛刺）”是否与“生产班次（白班、夜班）”有统计学上的关联。
    > 📌 **判断标准**：**P < 0.05** 代表两者存在**显著关联**（不独立）；P >= 0.05 代表互不影响。
    """)
    st.markdown("请以矩阵形式输入频数数据（每行代表一个类别，数字用空格隔开）。例如判断两台机器的缺陷类型分布是否一致。")
    matrix_in = st.text_area("输入频数矩阵 (例如2行3列):", "10 20 30\n15 15 35")
    
    if st.button("运行分析"):
        try:
            rows = matrix_in.strip().split('\n')
            matrix = [list(map(int, r.strip().split())) for r in rows if r.strip()]
            chi2, p_value, dof, expected = stats.chi2_contingency(matrix)
            st.write(f"- **卡方统计量**: {chi2:.2f} | **自由度**: {dof} | **P 值**: {p_value:.4f}")
            if p_value < 0.05: st.error("各分类变量之间存在显著相关性（拒绝独立性）。")
            else: st.success("各分类变量之间相互独立（无显著关联）。")
        except Exception as e:
            st.error("数据格式错误，请确保每行数字数量一致且均为整数。")

# ================= 7. 单因素 ANOVA =================
elif analysis_type == "7. 单因素方差分析 (One-Way ANOVA)":
    st.header("单因素 ANOVA & Tukey 事后检验")
    st.markdown("""
    > 🎯 **应用场景**：比较【3个或以上独立组别】的均值是否有显著差异。例如：对比3家不同压铸供应商交付零件的平均孔隙率。
    > 📌 **判断标准**：若 ANOVA **P < 0.05**，代表至少有一组与众不同。程序将**自动执行 Tukey 事后检验**，通过“字母分组法”帮您具体揪出是哪家供应商存在差异（同字母无差异，不同字母有差异）。
    """)
    c1, c2, c3 = st.columns(3)
    with c1: g1 = st.text_area("组 A:", "45 42 48 46 44")
    with c2: g2 = st.text_area("组 B:", "43 40 45 41 42")
    with c3: g3 = st.text_area("组 C:", "55 52 58 56 54")

    if st.button("运行分析"):
        d1, d2, d3 = parse_data(g1), parse_data(g2), parse_data(g3)
        if len(d1)>1 and len(d2)>1 and len(d3)>1:
            f_stat, p_value = stats.f_oneway(d1, d2, d3)
            st.write(f"- **ANOVA P 值**: {p_value:.4f}")
            if p_value < 0.05:
                st.error("至少有一组存在显著差异！自动执行 Tukey 检验：")
                all_vals = np.concatenate([d1, d2, d3])
                labels = ['A']*len(d1) + ['B']*len(d2) + ['C']*len(d3)
                tukey = pairwise_tukeyhsd(endog=all_vals, groups=labels, alpha=0.05)
                st.text(tukey.summary())
            else:
                st.success("各组均值无显著差异。")

# ================= 8. 双因素 ANOVA =================
elif analysis_type == "8. 双因素方差分析 (Two-Way ANOVA)":
    st.header("双因素 ANOVA (含交互作用评估)")
    st.markdown("""
    > 🎯 **应用场景**：同时评估【两个变量（因子）】对结果的影响，并寻找“1+1>2”的**交互作用**。例如：注塑时同时改变“压射速度”和“模具温度”，找出使风扇叶片强度达标的最佳参数组合。
    > 📌 **判断标准**：若交互项的 **P < 0.05**，代表存在显著的交互作用（特定参数搭配会产生质变效果）。
    """)
    st.markdown("需输入三列数据，长度必须一致。")
    
    col1, col2, col3 = st.columns(3)
    with col1: y_in = st.text_area("结果 Y (如强度):", "50 52 48 60 62 58 55 54 53 70 72 68")
    with col2: fa_in = st.text_area("因子 A (如速度):", "低 低 低 低 低 低 高 高 高 高 高 高")
    with col3: fb_in = st.text_area("因子 B (如温度):", "低 低 低 高 高 高 低 低 低 高 高 高")
    
    if st.button("运行分析"):
        y = parse_data(y_in)
        fa = parse_data(fa_in, dtype=str)
        fb = parse_data(fb_in, dtype=str)
        
        if len(y) == len(fa) == len(fb) and len(y) > 0:
            df = pd.DataFrame({'Y': y, 'FactorA': fa, 'FactorB': fb})
            try:
                model = ols('Y ~ C(FactorA) + C(FactorB) + C(FactorA):C(FactorB)', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.subheader("💡 ANOVA 分析表")
                st.dataframe(anova_table.style.format("{:.4f}"))
                
                p_a = anova_table.loc['C(FactorA)', 'PR(>F)']
                p_b = anova_table.loc['C(FactorB)', 'PR(>F)']
                p_ab = anova_table.loc['C(FactorA):C(FactorB)', 'PR(>F)']
                
                st.markdown("### 结论解读")
                st.write(f"- **因子 A 主效应**: P={p_a:.4f} ({'显著' if p_a<0.05 else '不显著'})")
                st.write(f"- **因子 B 主效应**: P={p_b:.4f} ({'显著' if p_b<0.05 else '不显著'})")
                if p_ab < 0.05:
                    st.error(f"⚠️ **发现显著交互作用！** (P={p_ab:.4f})。因子 A 和 B 搭配在一起产生了意想不到的效果。")
                else:
                    st.success(f"因子 A 和 B 之间无显著交互作用 (P={p_ab:.4f})。")
                    
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.pointplot(data=df, x='FactorA', y='Y', hue='FactorB', ax=ax, markers=['o', 's'], capsize=.1)
                ax.set_title("Interaction Plot")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"模型运算出错，请检查数据格式是否正确。错误信息：{e}")
        else:
            st.warning("三列数据的长度必须完全相等！请检查输入。")
