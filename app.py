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

# ================= 解决 Matplotlib 云端中文乱码问题 =================
# 优先使用 Linux 开源中文字体 (WenQuanYi Zen Hei)，向下兼容 Windows 和 Mac
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# ================= 页面配置 =================
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
    # 支持空格、逗号、换行符分隔的数据
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
            ax1.set_title("Histogram (直方图)")
            stats.probplot(data, dist="norm", plot=ax2)
            ax2.set_title("Normal Q-Q Plot (正态概率图)")
            st.pyplot(fig)

# ================= 1. 单样本 T检验 =================
elif analysis_type == "1. 单样本 T检验 (1-Sample t)":
    st.header("单样本 T检验")
    st.markdown("""
    > 🎯 **应用场景**：比较【当前的一批产品均值】与【固定的设计目标值】是否有显著偏差。
    > 📌 **判断标准**：**P < 0.05** 代表现状均值与目标值存在统计学上的**显著差异**；P >= 0.05 代表偏差属于正常波动范围。
    """)
    
    target_mean = st.number_input("输入目标均值:", value=3000.0)
    data_input = st.text_area("粘贴测试数据:", "2985 2990 3005 2980 2995 3010 2990 3000 2985 2995")
    
    if st.button("运行分析"):
        data = parse_data(data_input)
        if len(data) >= 2:
            t_stat, p_value = stats.ttest_1samp(data, target_mean)
            st.write(f"- **样本均值**: {np.mean(data):.2f} | **P 值**: {p_value:.4f}")
            if p_value < 0.05: st.error("结论：显著偏离目标值。")
            else: st.success("结论：属正常波动，未显著偏离目标值。")

# ================= 2. 双样本 T检验 =================
elif analysis_type == "2. 双样本 T检验 (2-Sample t)":
    st.header("双样本 T检验 (独立样本)")
    st.markdown("""
    > 🎯 **应用场景**：比较【两批相互独立的产品】的均值是否有显著差异。例如：对比验证“新工艺”的拉拔力是否真的比“旧工艺”高。
    > 📌 **判断标准**：**P < 0.05** 代表两组数据存在**显著差异**；P >= 0.05 代表两组处于同一水平。
    """)
    
    col1, col2 = st.columns(2)
    with col1: d1_in = st.text_area("组别 A (如旧工艺):", "15.2 14.8 15.5 14.9 15.1 15.3 14.7")
    with col2: d2_in = st.text_area("组别 B (如新工艺):", "16.1 15.9 16.5 16.2 16.0 16.3 16.1")
    
    if st.button("运行分析"):
        d1, d2 = parse_data(d1_in), parse_data(d2_in)
        if len(d1) >= 2 and len(d2) >= 2:
            _, p_lev = stats.levene(d1, d2)
            t_stat, p_value = stats.ttest_ind(d1, d2, equal_var=(p_lev > 0.05))
            st.write(f"- **方差齐性 P值**: {p_lev:.4f} (方差{'相等' if p_lev > 0.05 else '不相等'}) | **T检验 P值**: {p_value:.4f}")
            if p_value < 0.05: st.error("结论：两组数据存在显著差异！")
            else: st.success("结论：两组数据无显著差异。")
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=[d1, d2], ax=ax, palette="Set2")
            ax.set_xticklabels(['组别 A', '组别 B'])
            ax.set_title("双样本对比箱线图")
            st.pyplot(fig)

# ================= 3. 配对 T检验 =================
elif analysis_type == "3. 配对 T检验 (Paired t)":
    st.header("配对 T检验 (Paired t-test)")
    st.markdown("""
    > 🎯 **应用场景**：比较【同一批对象】在受到某种处理【前】与【后】的变化。两组数据必须是一一对应的。
    > 📌 **判断标准**：**P < 0.05** 代表处理前后发生了**显著变化**。
    """)
    col1, col2 = st.columns(2)
    with col1: d1_in = st.text_area("处理前 (Before):", "50 52 49 55 51 53 50")
    with col2: d2_in = st.text_area("处理后 (After):", "53 54 53 58 54 55 52")
    
    if st.button("运行分析"):
        d1, d2 = parse_data(d1_in), parse_data(d2_in)
        if len(d1) != len(d2) or len(d1) < 2:
            st.warning("两组数据数量必须完全相等且至少2个。")
        else:
            t_stat, p_value = stats.ttest_rel(d1, d2)
            st.write(f"- **差异均值 (后-前)**: {np.mean(np.array(d2)-np.array(d1)):.2f} | **P 值**: {p_value:.4f}")
            if p_value < 0.05: st.error("结论：处理前后存在显著差异！")
            else: st.success("结论：处理前后无显著差异。")

# ================= 4. 单比例检验 =================
elif analysis_type == "4. 单比例检验 (1-Proportion)":
    st.header("单比例检验")
    st.markdown("""
    > 🎯 **应用场景**：比较【当前的实际不良率】与【历史设定的目标不良率】是否有显著差异。
    > 📌 **判断标准**：**P < 0.05** 代表实际不良率与目标值存在**显著偏差**。
    """)
    target_p = st.number_input("目标不良率/比率 (0~1):", value=0.02)
    col1, col2 = st.columns(2)
    with col1: count = st.number_input("发现的不良数:", value=15, step=1)
    with col2: nobs = st.number_input("抽检总数:", value=500, step=1)
    
    if st.button("运行分析"):
        stat, p_value = proportions_ztest(count, nobs, value=target_p)
        st.write(f"- **实际不良率**: {count/nobs:.4f} | **P 值**: {p_value:.4f}")
        if p_value < 0.05: st.error("结论：实际比率与目标比率有显著差异！")
        else: st.success("结论：实际比率与目标比率无显著差异。")

# ================= 5. 双比例检验 =================
elif analysis_type == "5. 双比例检验 (2-Proportion)":
    st.header("双比例检验")
    st.markdown("""
    > 🎯 **应用场景**：比较【两个不同群体】的不良率是否有显著差异。例如：对比两家供应商交货的不合格品率。
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
        if p_value < 0.05: st.error("结论：两组不良率存在显著差异！")
        else: st.success("结论：两组不良率无显著差异。")

# ================= 6. 卡方检验 =================
elif analysis_type == "6. 卡方检验 (Chi-Square)":
    st.header("卡方检验 (独立性检验)")
    st.markdown("""
    > 🎯 **应用场景**：判断两个【分类事件】之间是否相互关联。例如：分析“缺陷类型”是否与“生产班次”有统计学关联。
    > 📌 **判断标准**：**P < 0.05** 代表两者存在**显著关联**（不独立）；P >= 0.05 代表互不影响。
    """)
    st.markdown("请以矩阵形式输入频数数据（每行代表一个类别，数字用空格隔开）。")
    matrix_in = st.text_area("输入频数矩阵 (例如2行3列):", "10 20 30\n15 15 35")
    
    if st.button("运行分析"):
        try:
            rows = matrix_in.strip().split('\n')
            matrix = [list(map(int, r.strip().split())) for r in rows if r.strip()]
            chi2, p_value, dof, expected = stats.chi2_contingency(matrix)
            st.write(f"- **卡方统计量**: {chi2:.2f} | **自由度**: {dof} | **P 值**: {p_value:.4f}")
            if p_value < 0.05: st.error("结论：各分类变量之间存在显著相关性（拒绝独立性）。")
            else: st.success("结论：各分类变量之间相互独立（无显著关联）。")
        except Exception as e:
            st.error("数据格式错误，请确保每行数字数量一致且均为整数。")

# ================= 7. 单因素 ANOVA =================
elif analysis_type == "7. 单因素方差分析 (One-Way ANOVA)":
    st.header("单因素 ANOVA & Tukey 事后检验")
    st.markdown("""
    > 🎯 **应用场景**：比较【3个或以上独立组别】的均值是否有显著差异。例如：对比3家不同供应商交付零件的关键参数。
    > 📌 **判断标准**：若 ANOVA **P < 0.05**，代表至少有一组与众不同。程序将**自动执行 Tukey 事后检验**，帮您具体揪出差异方（看表格的 reject 列，True 代表有显著差异）。
    """)
    c1, c2, c3 = st.columns(3)
    with c1: g1 = st.text_area("组 A:", "45 42 48 46 44 47")
    with c2: g2 = st.text_area("组 B:", "43 40 45 41 42 41")
    with c3: g3 = st.text_area("组 C:", "55 52 58 56 54 57")

    if st.button("运行分析"):
        d1, d2, d3 = parse_data(g1), parse_data(g2), parse_data(g3)
        if len(d1)>1 and len(d2)>1 and len(d3)>1:
            f_stat, p_value = stats.f_oneway(d1, d2, d3)
            st.write(f"- **ANOVA P 值**: {p_value:.4f}")
            if p_value < 0.05:
                st.error("结论：至少有一组存在显著差异！自动执行 Tukey 检验：")
                all_vals = np.concatenate([d1, d2, d3])
                labels = ['A']*len(d1) + ['B']*len(d2) + ['C']*len(d3)
                tukey = pairwise_tukeyhsd(endog=all_vals, groups=labels, alpha=0.05)
                st.text(tukey.summary())
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=[d1, d2, d3], ax=ax, palette="Set1")
                ax.set_xticklabels(['组别 A', '组别 B', '组别 C'])
                ax.set_title("ANOVA 分组箱线图")
                st.pyplot(fig)
            else:
                st.success("结论：各组均值无显著差异。")

# ================= 8. 双因素 ANOVA =================
elif analysis_type == "8. 双因素方差分析 (Two-Way ANOVA)":
    st.header("双因素 ANOVA (含交互作用评估)")
    st.markdown("""
    > 🎯 **应用场景**：同时评估【两个变量（因子）】对结果的影响，并寻找“1+1>2”的**交互作用**。
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
                    
                # 绘制交互作用图
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.pointplot(data=df, x='FactorA', y='Y', hue='FactorB', ax=ax, markers=['o', 's'], capsize=.1)
                ax.set_title("交互作用图 (Interaction Plot)")
                st.pyplot(fig)
                
                # ================= 🌟 图表诊断与大白话解读向导 🌟 =================
                st.markdown("---")
                st.subheader("📊 交互作用图深入解读向导")

                if p_ab < 0.05:
                    interpretation = """
                    **🎯 图表诊断：发现显著的“化学反应” (线段非平行或交叉)**

                    当交互作用显著时，意味着这两个因子**互相制约**。我们不能再孤立地谈论某个因子的好坏，而是要寻找**最佳的特定参数搭配**。

                    **👀 应该怎么看图？**

                    1.  **看趋势差异**：请特别观察不同颜色线段从左到右的**斜率差异**。它们不仅高度不同，趋势也不同。例如，在低 FactorA 下，FactorB 带来的提升可能较小，但在高 FactorA 下，FactorB 的提升效果可能会瞬间翻倍（或者反过来下降）。
                    2.  **寻找黄金组合**：既然有了化学反应，我们的目的就是寻找使结果 Y 达到期望值（比如让强度最大化、孔隙率最小化）的**特定坐标点**。
                        * 如果目的是让 Y **最大**：在图中找到最高点对应的横坐标 (Factor A) 和图例颜色 (Factor B)。
                        * 如果目的是让 Y **最小**：找到图中的最低点对应的特定搭配。

                    > **💡 工艺启示**：调整工艺参数时，必须同时绑定这两个因子，不能“顾头不顾尾”。
                    """
                    st.info(interpretation)
                else:
                    interpretation = """
                    **🎯 图表诊断：无显著交互作用 (线段大致平行)**

                    当交互作用不显著时，意味着这两个因子**各干各的**，互不干扰。我们可以分别得出独立的工艺优化结论：

                    **👀 应该怎么看图？**

                    1.  **因子 A 的单独影响**：请观察线段的**整体倾斜度**。如果所有线段都从左向右显著**上升**，说明增加 FactorA 都会稳定提高结果 Y，不论 FactorB 处于什么状态。
                    2.  **因子 B 的单独影响**：请观察两条线段之间的**垂直距离**。如果一条线一直显著高于另一条线，说明将 FactorB 设定在该状态下，都能产生稳定的提升效果。
                    3.  **最终优化思路**：如果想让 Y 最大，只需要简单粗暴地将这两个因子都分别调至它们各自能产生最高 Y 值的水平即可（通常是右上角区域），不需要考虑复杂的搭配。
                    """
                    st.info(interpretation)
                
            except Exception as e:
                st.error(f"模型运算出错，请检查数据格式是否正确。错误信息：{e}")
        else:
            st.warning("三列数据的长度必须完全相等！请检查输入。")
