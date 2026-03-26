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
        "8. 双因素方差分析 (Two-Way ANOVA)",
        "9. 测量系统分析 (MSA Gage R&R)"
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

# ================= 0-6 基础检验模块 (保持原样) =================
if analysis_type == "0. 正态性检验 (Normality Test)":
    st.header("正态性检验 (Shapiro-Wilk Test)")
    st.markdown("> 🎯 **前提要求**：T检验、ANOVA 等连续型数据分析工具，均要求数据服从正态分布。**P > 0.05** 代表正态。")
    data_input = st.text_area("粘贴测试数据:", "10.1 9.8 10.5 10.0 10.2 9.9 10.3 10.1 9.7 10.4")
    if st.button("运行正态体检"):
        data = parse_data(data_input)
        if len(data) >= 3:
            stat, p_value = stats.shapiro(data)
            st.write(f"- **P 值**: {p_value:.4f}")
            if p_value > 0.05: st.success("✅ 数据服从正态分布。")
            else: st.error("❌ 数据不服从正态分布。")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(data, kde=True, ax=ax1)
            stats.probplot(data, dist="norm", plot=ax2)
            st.pyplot(fig)

elif analysis_type == "1. 单样本 T检验 (1-Sample t)":
    st.header("单样本 T检验")
    st.markdown("> 🎯 **应用场景**：比较【当前的一批产品均值】与【固定的设计目标值】是否有显著偏差。")
    target_mean = st.number_input("输入目标均值:", value=3000.0)
    data_input = st.text_area("粘贴测试数据:", "2985 2990 3005 2980 2995 3010 2990 3000 2985 2995")
    if st.button("运行分析"):
        data = parse_data(data_input)
        if len(data) >= 2:
            t_stat, p_value = stats.ttest_1samp(data, target_mean)
            st.write(f"- **样本均值**: {np.mean(data):.2f} | **P 值**: {p_value:.4f}")
            if p_value < 0.05: st.error("结论：显著偏离目标值。")
            else: st.success("结论：属正常波动，未显著偏离目标值。")

elif analysis_type == "2. 双样本 T检验 (2-Sample t)":
    st.header("双样本 T检验 (独立样本)")
    st.markdown("> 🎯 **应用场景**：比较【两批相互独立的产品】的均值是否有显著差异。")
    col1, col2 = st.columns(2)
    with col1: d1_in = st.text_area("组别 A (如旧工艺):", "15.2 14.8 15.5 14.9 15.1 15.3 14.7")
    with col2: d2_in = st.text_area("组别 B (如新工艺):", "16.1 15.9 16.5 16.2 16.0 16.3 16.1")
    if st.button("运行分析"):
        d1, d2 = parse_data(d1_in), parse_data(d2_in)
        if len(d1) >= 2 and len(d2) >= 2:
            _, p_lev = stats.levene(d1, d2)
            t_stat, p_value = stats.ttest_ind(d1, d2, equal_var=(p_lev > 0.05))
            st.write(f"- **方差齐性 P值**: {p_lev:.4f} | **T检验 P值**: {p_value:.4f}")
            if p_value < 0.05: st.error("结论：两组数据存在显著差异！")
            else: st.success("结论：两组数据无显著差异。")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=[d1, d2], ax=ax, palette="Set2")
            ax.set_xticklabels(['组别 A', '组别 B'])
            st.pyplot(fig)

elif analysis_type == "3. 配对 T检验 (Paired t)":
    st.header("配对 T检验 (Paired t-test)")
    st.markdown("> 🎯 **应用场景**：比较【同一批对象】在受到某种处理【前】与【后】的变化。")
    col1, col2 = st.columns(2)
    with col1: d1_in = st.text_area("处理前 (Before):", "50 52 49 55 51 53 50")
    with col2: d2_in = st.text_area("处理后 (After):", "53 54 53 58 54 55 52")
    if st.button("运行分析"):
        d1, d2 = parse_data(d1_in), parse_data(d2_in)
        if len(d1) == len(d2) and len(d1) >= 2:
            t_stat, p_value = stats.ttest_rel(d1, d2)
            st.write(f"- **差异均值 (后-前)**: {np.mean(np.array(d2)-np.array(d1)):.2f} | **P 值**: {p_value:.4f}")
            if p_value < 0.05: st.error("结论：处理前后存在显著差异！")
            else: st.success("结论：处理前后无显著差异。")

elif analysis_type == "4. 单比例检验 (1-Proportion)":
    st.header("单比例检验")
    target_p = st.number_input("目标不良率/比率 (0~1):", value=0.02)
    col1, col2 = st.columns(2)
    with col1: count = st.number_input("发现的不良数:", value=15, step=1)
    with col2: nobs = st.number_input("抽检总数:", value=500, step=1)
    if st.button("运行分析"):
        stat, p_value = proportions_ztest(count, nobs, value=target_p)
        st.write(f"- **实际不良率**: {count/nobs:.4f} | **P 值**: {p_value:.4f}")
        if p_value < 0.05: st.error("结论：实际比率与目标比率有显著差异！")
        else: st.success("结论：实际比率与目标比率无显著差异。")

elif analysis_type == "5. 双比例检验 (2-Proportion)":
    st.header("双比例检验")
    col1, col2 = st.columns(2)
    with col1:
        c1 = st.number_input("组A 不良数:", value=12, step=1)
        n1 = st.number_input("组A 抽样数:", value=1000, step=1)
    with col2:
        c2 = st.number_input("组B 不良数:", value=5, step=1)
        n2 = st.number_input("组B 抽样数:", value=800, step=1)
    if st.button("运行分析"):
        stat, p_value = proportions_ztest(np.array([c1, c2]), np.array([n1, n2]))
        st.write(f"- **组A不良率**: {c1/n1:.4f} | **组B不良率**: {c2/n2:.4f} | **P 值**: {p_value:.4f}")
        if p_value < 0.05: st.error("结论：两组不良率存在显著差异！")
        else: st.success("结论：两组不良率无显著差异。")

elif analysis_type == "6. 卡方检验 (Chi-Square)":
    st.header("卡方检验 (独立性检验)")
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
        except Exception:
            st.error("数据格式错误。")

# ================= 7. 单因素 ANOVA =================
elif analysis_type == "7. 单因素方差分析 (One-Way ANOVA)":
    st.header("单因素 ANOVA & Tukey 事后检验")
    st.markdown("> 🎯 **应用场景**：比较【3个或以上独立组别】的均值是否有显著差异。")
    num_groups = st.number_input("请选择要比较的组数 (2 到 10 组):", min_value=2, max_value=10, value=3, step=1)
    groups_data_inputs = []
    group_names = []
    cols = st.columns(3)
    for i in range(num_groups):
        group_name = chr(ord('A') + i)
        with cols[i % 3]:
            default_text = ""
            if i == 0: default_text = "45 42 48 46 44 47"
            elif i == 1: default_text = "43 40 45 41 42 41"
            elif i == 2: default_text = "55 52 58 56 54 57"
            text_input = st.text_area(f"组别 {group_name}:", value=default_text, key=f"anova_{i}")
            groups_data_inputs.append(text_input)
            group_names.append(group_name)

    if st.button("运行分析"):
        parsed_data = []
        valid_labels = []
        for i, d_text in enumerate(groups_data_inputs):
            d = parse_data(d_text)
            if len(d) > 1:
                parsed_data.append(d)
                valid_labels.append(group_names[i])
        if len(parsed_data) >= 2:
            f_stat, p_value = stats.f_oneway(*parsed_data)
            st.write(f"- **ANOVA P 值**: {p_value:.4f}")
            if p_value < 0.05:
                st.error("结论：至少有一组存在显著差异！自动执行 Tukey 检验：")
                all_vals = np.concatenate(parsed_data)
                labels = []
                for name, data in zip(valid_labels, parsed_data): labels.extend([name] * len(data))
                tukey = pairwise_tukeyhsd(endog=all_vals, groups=labels, alpha=0.05)
                st.text(tukey.summary())
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=parsed_data, ax=ax, palette="Set1")
                ax.set_xticklabels([f"组别 {name}" for name in valid_labels])
                st.pyplot(fig)
            else: st.success("结论：各组均值无显著差异。")

# ================= 8. 双因素 ANOVA =================
elif analysis_type == "8. 双因素方差分析 (Two-Way ANOVA)":
    st.header("双因素 ANOVA (含交互作用评估)")
    st.markdown("> 🎯 **应用场景**：同时评估【两个变量（因子）】对结果的影响，并寻找交互作用。")
    col1, col2, col3 = st.columns(3)
    with col1: y_in = st.text_area("结果 Y (如强度):", "50 52 48 60 62 58 55 54 53 70 72 68")
    with col2: fa_in = st.text_area("因子 A (如速度):", "低 低 低 低 低 低 高 高 高 高 高 高")
    with col3: fb_in = st.text_area("因子 B (如温度):", "低 低 低 高 高 高 低 低 低 高 高 高")
    if st.button("运行分析"):
        y, fa, fb = parse_data(y_in), parse_data(fa_in, dtype=str), parse_data(fb_in, dtype=str)
        if len(y) == len(fa) == len(fb) and len(y) > 0:
            df = pd.DataFrame({'Y': y, 'FactorA': fa, 'FactorB': fb})
            try:
                model = ols('Y ~ C(FactorA) + C(FactorB) + C(FactorA):C(FactorB)', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.dataframe(anova_table.style.format("{:.4f}"))
                p_a = anova_table.loc['C(FactorA)', 'PR(>F)']
                p_b = anova_table.loc['C(FactorB)', 'PR(>F)']
                p_ab = anova_table.loc['C(FactorA):C(FactorB)', 'PR(>F)']
                
                if p_ab < 0.05: st.error(f"⚠️ **发现显著交互作用！** (P={p_ab:.4f})")
                else: st.success(f"无显著交互作用 (P={p_ab:.4f})")
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.pointplot(data=df, x='FactorA', y='Y', hue='FactorB', ax=ax, markers=['o', 's'], capsize=.1)
                st.pyplot(fig)
            except Exception as e: st.error(f"模型运算出错: {e}")

# ================= 9. 测量系统分析 (MSA Gage R&R) =================
elif analysis_type == "9. 测量系统分析 (MSA Gage R&R)":
    st.header("测量系统分析 (Gage R&R - 交叉 ANOVA 法)")
    st.markdown("""
    > 🎯 **应用场景**：评估测量系统的重复性（设备误差 EV）和再现性（人为误差 AV）。例如：张三、李四和王五 3 个检验员用同一把千分尺，测量 10 个电机轴径，每个人各测 3 次。
    > 📌 **判断标准**：
    > * **%GRR (量具变差占比)**：$< 10\%$ 卓越；$10\% \sim 30\%$ 条件接受；$> 30\%$ 拒绝，量具不合格。
    > * **NDC (可区分的类别数)**：必须 $\ge 5$。如果小于 5，说明仪器分辨率太低，只能当通止规用。
    """)
    st.markdown("⚠️ **数据输入说明**：请确保三列数据的行数完全一致（每行代表一次独立测量记录）。")

    col1, col2, col3 = st.columns(3)
    with col1: 
        part_in = st.text_area("零件编号 (Part):", "1 1 2 2 3 3 1 1 2 2 3 3")
    with col2: 
        op_in = st.text_area("检验员 (Appraiser):", "张三 张三 张三 张三 张三 张三 李四 李四 李四 李四 李四 李四")
    with col3: 
        val_in = st.text_area("测量数值 (Value):", "15.1 15.0 16.2 16.1 14.8 14.9 15.2 15.1 16.0 16.2 14.9 14.7")
        
    if st.button("🚀 运行 MSA 分析"):
        part = parse_data(part_in, dtype=str)
        appraiser = parse_data(op_in, dtype=str)
        value = parse_data(val_in)
        
        if len(part) == len(appraiser) == len(value) and len(value) > 0:
            df = pd.DataFrame({'Value': value, 'Part': part, 'Appraiser': appraiser})
            
            try:
                # 建立方差分析模型提取均方差 (Mean Squares)
                model = ols('Value ~ C(Part) + C(Appraiser) + C(Part):C(Appraiser)', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                # 获取各个维度的数量
                a = df['Appraiser'].nunique()
                p = df['Part'].nunique()
                n = len(df) / (a * p) # 测量次数 (Trials)
                
                if n <= 1:
                    st.error("每个检验员测每个零件至少需要测 2 次以上（有重复测试）才能计算出设备误差 EV！")
                else:
                    # 提取均方 (MS)
                    ms_p = anova_table.loc['C(Part)', 'mean_sq']
                    ms_a = anova_table.loc['C(Appraiser)', 'mean_sq']
                    ms_pa = anova_table.loc['C(Part):C(Appraiser)', 'mean_sq']
                    ms_e = anova_table.loc['Residual', 'mean_sq']
                    
                    # 计算方差分量 (Variance Components) AIAG 标准逻辑
                    var_e = ms_e  # EV 方差
                    var_pa = max(0, (ms_pa - ms_e) / n) # 交互作用方差
                    var_a = max(0, (ms_a - ms_pa) / (p * n)) # AV 方差
                    var_p = max(0, (ms_p - ms_pa) / (a * n)) # PV 方差
                    
                    # 提取标准差分量 (Standard Deviation)
                    ev = np.sqrt(var_e)
                    av = np.sqrt(var_a + var_pa) # 将交互作用合并入人为误差
                    grr = np.sqrt(ev**2 + av**2) # 总测量系统变差
                    pv = np.sqrt(var_p) # 零件间变差
                    tv = np.sqrt(grr**2 + pv**2) # 总变差 Total Variation
                    
                    # 计算核心指标
                    pct_grr = (grr / tv) * 100
                    pct_ev = (ev / tv) * 100
                    pct_av = (av / tv) * 100
                    pct_pv = (pv / tv) * 100
                    ndc = max(1, int(np.floor(1.41 * (pv / grr))))
                    
                    # 结果展示
                    st.subheader("💡 MSA 核心诊断报告")
                    
                    # 关键指标卡片
                    col_met1, col_met2 = st.columns(2)
                    col_met1.metric(label="总 Gage R&R (%)", value=f"{pct_grr:.2f}%", delta="< 10% 卓越 | < 30% 及格" if pct_grr < 30 else "> 30% 需整改", delta_color="inverse")
                    col_met2.metric(label="NDC (可区分类别数)", value=ndc, delta=">= 5 达标" if ndc >= 5 else "< 5 不达标")
                    
                    if pct_grr < 10: st.success("✅ 结论：系统非常卓越，可直接用于关键特性的管控！")
                    elif pct_grr < 30: st.warning("⚠️ 结论：系统处于边缘地带，根据测量成本和特性重要性有条件接受。")
                    else: st.error("❌ 结论：系统完全不可接受！误差吃掉了太多公差。必须检查设备松动或重新规范检验员手法！")

                    # 变差拆解明细
                    st.markdown("### 🔍 误差来源拆解 (谁背锅？)")
                    st.write(f"- **总测量系统误差 (GRR)**: {pct_grr:.2f}%")
                    st.write(f"  - ➡️ **设备变差 / 重复性 (EV)**: {pct_ev:.2f}% *(设备老旧或松动)*")
                    st.write(f"  - ➡️ **人为变差 / 再现性 (AV)**: {pct_av:.2f}% *(检验员测量手法不统一)*")
                    st.write(f"- **真实产品波动 (PV)**: {pct_pv:.2f}%")

                    # 变差分量堆叠条形图可视化
                    st.markdown("### 📊 变差分布图")
                    fig, ax = plt.subplots(figsize=(8, 3))
                    components = ['%EV (设备)', '%AV (人为)', '%GRR (总误差)', '%PV (零件真实波动)']
                    values = [pct_ev, pct_av, pct_grr, pct_pv]
                    sns.barplot(x=values, y=components, palette="Blues_d", ax=ax)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('占总变差百分比 (%)')
                    for i, v in enumerate(values):
                        ax.text(v + 1, i, f"{v:.1f}%", va='center')
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"模型运算出错，请确保数据输入格式正确。注意：ANOVA 模型要求数据交叉平衡（所有人测相同的零件）。错误信息：{e}")
        else:
            st.warning("三列数据的长度必须完全相等！请检查输入。")
