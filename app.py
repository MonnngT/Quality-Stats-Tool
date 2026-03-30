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
# 优先使用 Linux 开源中文字体，向下兼容 Windows 和 Mac
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# ================= 页面配置 =================
st.set_page_config(page_title="全能版质量统计中心", layout="wide")
st.title("📊 Web 版 Minitab - 全能质量统计工具")
st.markdown("💡 **操作提示**：下方所有数据表格，均支持直接从本地 **Excel** 中选中整列复制，并在网页表格的表头处按 `Ctrl+V` 粘贴！")

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
        "9. 测量系统分析 (MSA Gage R&R)",
        "10. 过程能力分析 (Cp/Cpk)"
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
    > 🎯 **前提要求**：T检验、ANOVA 等连续型数据分析工具，均要求数据服从正态分布。
    > 📌 **判断标准**：**P > 0.05** 代表数据为正态分布，可以放心使用后续高级工具；P < 0.05 代表数据偏态。
    """)
    
    df_default = pd.DataFrame({"测试数据": [10.1, 9.8, 10.5, 10.0, 10.2, 9.9, 10.3, 10.1, 9.7, 10.4]})
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
    
    if st.button("运行正态体检"):
        data = pd.to_numeric(edited_df["测试数据"], errors='coerce').dropna().tolist()
        if len(data) < 3:
            st.warning("至少需要输入 3 个有效数据点。")
        else:
            stat, p_value = stats.shapiro(data)
            st.subheader("💡 检验结果")
            st.write(f"- **P 值**: {p_value:.4f}")
            if p_value > 0.05:
                st.success("✅ P > 0.05，接受原假设：数据**服从**正态分布。")
            else:
                st.error("❌ P < 0.05，拒绝原假设：数据**不服从**正态分布。")
            
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
    
    df_default = pd.DataFrame({"测试数据": [2985, 2990, 3005, 2980, 2995, 3010, 2990, 3000, 2985, 2995]})
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
    
    if st.button("运行分析"):
        data = pd.to_numeric(edited_df["测试数据"], errors='coerce').dropna().tolist()
        if len(data) >= 2:
            t_stat, p_value = stats.ttest_1samp(data, target_mean)
            st.write(f"- **样本均值**: {np.mean(data):.2f} | **P 值**: {p_value:.4f}")
            if p_value < 0.05: st.error("结论：显著偏离目标值。")
            else: st.success("结论：属正常波动，未显著偏离目标值。")

# ================= 2. 双样本 T检验 =================
elif analysis_type == "2. 双样本 T检验 (2-Sample t)":
    st.header("双样本 T检验 (独立样本)")
    st.markdown("""
    > 🎯 **应用场景**：比较【两批相互独立的产品】的均值是否有显著差异。例如对比新旧工艺。
    > 📌 **判断标准**：**P < 0.05** 代表两组数据存在**显著差异**。
    """)
    df_default = pd.DataFrame({
        "组别 A (如旧工艺)": [15.2, 14.8, 15.5, 14.9, 15.1, 15.3, 14.7],
        "组别 B (如新工艺)": [16.1, 15.9, 16.5, 16.2, 16.0, 16.3, 16.1]
    })
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
    
    if st.button("运行分析"):
        d1 = pd.to_numeric(edited_df["组别 A (如旧工艺)"], errors='coerce').dropna().tolist()
        d2 = pd.to_numeric(edited_df["组别 B (如新工艺)"], errors='coerce').dropna().tolist()
        if len(d1) >= 2 and len(d2) >= 2:
            _, p_lev = stats.levene(d1, d2)
            t_stat, p_value = stats.ttest_ind(d1, d2, equal_var=(p_lev > 0.05))
            st.write(f"- **方差齐性 P值**: {p_lev:.4f} (方差{'相等' if p_lev > 0.05 else '不相等'}) | **T检验 P值**: {p_value:.4f}")
            if p_value < 0.05: st.error("结论：两组数据存在显著差异！")
            else: st.success("结论：两组数据无显著差异。")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=[d1, d2], ax=ax, palette="Set2")
            ax.set_xticklabels(['组别 A', '组别 B'])
            st.pyplot(fig)

# ================= 3. 配对 T检验 =================
elif analysis_type == "3. 配对 T检验 (Paired t)":
    st.header("配对 T检验 (Paired t-test)")
    st.markdown("""
    > 🎯 **应用场景**：比较【同一批对象】在受到某种处理【前】与【后】的变化。两组数据必须一一对应。
    > 📌 **判断标准**：**P < 0.05** 代表处理前后发生了**显著变化**。
    """)
    df_default = pd.DataFrame({
        "处理前 (Before)": [50, 52, 49, 55, 51, 53, 50],
        "处理后 (After)": [53, 54, 53, 58, 54, 55, 52]
    })
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
    
    if st.button("运行分析"):
        d1 = pd.to_numeric(edited_df["处理前 (Before)"], errors='coerce').dropna().tolist()
        d2 = pd.to_numeric(edited_df["处理后 (After)"], errors='coerce').dropna().tolist()
        if len(d1) != len(d2) or len(d1) < 2:
            st.warning("两组有效数据数量必须完全相等且至少2个。")
        else:
            t_stat, p_value = stats.ttest_rel(d1, d2)
            st.write(f"- **差异均值 (后-前)**: {np.mean(np.array(d2)-np.array(d1)):.2f} | **P 值**: {p_value:.4f}")
            if p_value < 0.05: st.error("结论：处理前后存在显著差异！")
            else: st.success("结论：处理前后无显著差异。")

# ================= 4. 单比例检验 =================
elif analysis_type == "4. 单比例检验 (1-Proportion)":
    st.header("单比例检验")
    st.markdown("> 🎯 **应用场景**：比较【当前的实际不良率】与【设定的目标不良率】是否有显著差异。")
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
    st.markdown("> 🎯 **应用场景**：比较【两个不同群体】的不良率是否有显著差异。例如对比两家供应商。")
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

# ================= 6. 卡方检验 =================
elif analysis_type == "6. 卡方检验 (Chi-Square)":
    st.header("卡方检验 (独立性检验)")
    st.markdown("> 🎯 **应用场景**：判断两个【分类事件】之间是否相互关联。例如“缺陷类型”与“生产班次”。")
    st.markdown("请以矩阵形式输入频数数据（每行代表一个类别，数字用空格隔开）。")
    matrix_in = st.text_area("输入频数矩阵 (例如2行3列):", "10 20 30\n15 15 35")
    if st.button("运行分析"):
        try:
            rows = matrix_in.strip().split('\n')
            matrix = [list(map(int, r.strip().split())) for r in rows if r.strip()]
            chi2, p_value, dof, expected = stats.chi2_contingency(matrix)
            st.write(f"- **卡方统计量**: {chi2:.2f} | **自由度**: {dof} | **P 值**: {p_value:.4f}")
            if p_value < 0.05: st.error("结论：存在显著相关性（拒绝独立性）。")
            else: st.success("结论：各分类变量之间相互独立（无显著关联）。")
        except Exception:
            st.error("数据格式错误。")

# ================= 7. 单因素 ANOVA =================
elif analysis_type == "7. 单因素方差分析 (One-Way ANOVA)":
    st.header("单因素 ANOVA & Tukey 事后检验")
    st.markdown("""
    > 🎯 **应用场景**：比较【3个或以上独立组别】的均值是否有显著差异。
    > 📌 **判断标准**：若 ANOVA **P < 0.05**，代表至少有一组与众不同。程序将**自动执行 Tukey 事后检验**揪出差异方。
    """)
    num_groups = st.number_input("请选择要比较的组数 (2 到 10 组):", min_value=2, max_value=10, value=3, step=1)
    
    cols = [f"组别 {chr(ord('A') + i)}" for i in range(num_groups)]
    default_data = {}
    for i, col in enumerate(cols):
        if i == 0: default_data[col] = [45.1, 42.5, 48.0, 46.2, 44.8, 47.1]
        elif i == 1: default_data[col] = [43.0, 40.5, 45.1, 41.2, 42.1, 41.5]
        elif i == 2: default_data[col] = [55.2, 52.8, 58.1, 56.4, 54.9, 57.0]
        else: default_data[col] = [np.nan] * 6

    df_default = pd.DataFrame(default_data)
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)

    if st.button("运行分析"):
        parsed_data = []
        valid_labels = []
        for col in cols:
            d = pd.to_numeric(edited_df[col], errors='coerce').dropna().tolist()
            if len(d) > 1:
                parsed_data.append(d)
                valid_labels.append(col)
        
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
                ax.set_xticklabels(valid_labels)
                st.pyplot(fig)
            else: st.success("结论：各组均值无显著差异。")
        else:
            st.warning("至少需要 2 个有效的组别且每组至少 2 个数据。")

# ================= 8. 双因素 ANOVA =================
elif analysis_type == "8. 双因素方差分析 (Two-Way ANOVA)":
    st.header("双因素 ANOVA (含交互作用评估)")
    st.markdown("""
    > 🎯 **应用场景**：同时评估【两个变量（因子）】对结果的影响，并寻找“1+1>2”的**交互作用**。
    > 📌 **判断标准**：若交互项的 **P < 0.05**，代表存在显著的交互作用。
    """)
    df_default = pd.DataFrame({
        "结果 Y (如强度)": [50, 52, 48, 60, 62, 58, 55, 54, 53, 70, 72, 68],
        "因子 A (如速度)": ["低", "低", "低", "低", "低", "低", "高", "高", "高", "高", "高", "高"],
        "因子 B (如温度)": ["低", "低", "低", "高", "高", "高", "低", "低", "低", "高", "高", "高"]
    })
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)

    if st.button("运行分析"):
        df = edited_df.dropna(how='any').copy()
        df['Y'] = pd.to_numeric(df['结果 Y (如强度)'], errors='coerce')
        df = df.dropna(subset=['Y'])
        df['FactorA'] = df['因子 A (如速度)'].astype(str)
        df['FactorB'] = df['因子 B (如温度)'].astype(str)

        if len(df) > 0:
            try:
                model = ols('Y ~ C(FactorA) + C(FactorB) + C(FactorA):C(FactorB)', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.subheader("💡 ANOVA 分析表")
                st.dataframe(anova_table.style.format("{:.4f}"))
                p_ab = anova_table.loc['C(FactorA):C(FactorB)', 'PR(>F)']
                
                if p_ab < 0.05: st.error(f"⚠️ **发现显著交互作用！** (P={p_ab:.4f})")
                else: st.success(f"无显著交互作用 (P={p_ab:.4f})")
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.pointplot(data=df, x='FactorA', y='Y', hue='FactorB', ax=ax, markers=['o', 's'], capsize=.1)
                st.pyplot(fig)
            except Exception as e: st.error(f"模型运算出错: {e}。请确保数据组合完整无缺失。")

# ================= 9. 测量系统分析 (MSA Gage R&R) =================
elif analysis_type == "9. 测量系统分析 (MSA Gage R&R)":
    st.header("测量系统分析 (Gage R&R - 交叉 ANOVA 法)")
    st.markdown("""
    > 🎯 **应用场景**：评估测量系统的重复性（设备误差 EV）和再现性（人为误差 AV）。
    > 📌 **判断标准**：
    > * **%GRR (量具变差占比)**：$< 10\%$ 卓越；$10\% \sim 30\%$ 条件接受；$> 30\%$ 拒绝，量具不合格。
    > * **NDC (可区分的类别数)**：必须 $\ge 5$。
    """)
    
    df_default = pd.DataFrame({
        "零件编号 (Part)": ["1", "1", "2", "2", "3", "3", "1", "1", "2", "2", "3", "3"],
        "检验员/量具 (Appraiser)": ["张三", "张三", "张三", "张三", "张三", "张三", "李四", "李四", "李四", "李四", "李四", "李四"],
        "测量数值 (Value)": [15.1, 15.0, 16.2, 16.1, 14.8, 14.9, 15.2, 15.1, 16.0, 16.2, 14.9, 14.7]
    })
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
        
    if st.button("🚀 运行 MSA 分析"):
        df = edited_df.dropna(how='any').copy()
        df['Value'] = pd.to_numeric(df['测量数值 (Value)'], errors='coerce')
        df = df.dropna(subset=['Value'])
        df['Part'] = df['零件编号 (Part)'].astype(str)
        df['Appraiser'] = df['检验员/量具 (Appraiser)'].astype(str)
        
        if len(df) > 0:
            counts = df.groupby(['Part', 'Appraiser']).size()
            if counts.min() < 2:
                st.error("❌ **数据错误：缺少重复测量数据！**\n\n交叉 ANOVA 要求【每位检验员/量具】对【每个零件】必须测量至少 **2 次** 才能算出设备误差 (EV)。")
            else:
                try:
                    model = ols('Value ~ C(Part) + C(Appraiser) + C(Part):C(Appraiser)', data=df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']
                    
                    a = df['Appraiser'].nunique()
                    p = df['Part'].nunique()
                    n = len(df) / (a * p) 
                    
                    ms_p = anova_table.loc['C(Part)', 'mean_sq']
                    ms_a = anova_table.loc['C(Appraiser)', 'mean_sq']
                    ms_pa = anova_table.loc['C(Part):C(Appraiser)', 'mean_sq']
                    ms_e = anova_table.loc['Residual', 'mean_sq']
                    
                    var_e = ms_e  
                    var_pa = max(0, (ms_pa - ms_e) / n) 
                    var_a = max(0, (ms_a - ms_pa) / (p * n)) 
                    var_p = max(0, (ms_p - ms_pa) / (a * n)) 
                    
                    ev = np.sqrt(var_e)
                    av = np.sqrt(var_a + var_pa) 
                    grr = np.sqrt(ev**2 + av**2) 
                    pv = np.sqrt(var_p) 
                    tv = np.sqrt(grr**2 + pv**2) 
                    
                    pct_grr = (grr / tv) * 100
                    pct_ev = (ev / tv) * 100
                    pct_av = (av / tv) * 100
                    pct_pv = (pv / tv) * 100
                    ndc = max(1, int(np.floor(1.41 * (pv / grr))))
                    
                    st.subheader("💡 MSA 核心诊断报告")
                    col_met1, col_met2 = st.columns(2)
                    col_met1.metric(label="总 Gage R&R (%)", value=f"{pct_grr:.2f}%", delta="< 10% 卓越 | < 30% 及格" if pct_grr < 30 else "> 30% 需整改", delta_color="inverse")
                    col_met2.metric(label="NDC (可区分类别数)", value=ndc, delta=">= 5 达标" if ndc >= 5 else "< 5 不达标")
                    
                    st.markdown("### 🔍 误差来源拆解")
                    st.write(f"- **总测量系统误差 (GRR)**: {pct_grr:.2f}%")
                    st.write(f"  - ➡️ **设备变差 (EV)**: {pct_ev:.2f}%")
                    st.write(f"  - ➡️ **人为变差 (AV)**: {pct_av:.2f}%")
                    st.write(f"- **真实产品波动 (PV)**: {pct_pv:.2f}%")

                    fig, ax = plt.subplots(figsize=(8, 3))
                    components = ['%EV (设备)', '%AV (人为)', '%GRR (总误差)', '%PV (零件真实波动)']
                    values = [pct_ev, pct_av, pct_grr, pct_pv]
                    sns.barplot(x=values, y=components, palette="Blues_d", ax=ax)
                    ax.set_xlim(0, 100)
                    for i, v in enumerate(values):
                        ax.text(v + 1, i, f"{v:.2f}%", va='center')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"模型运算出错，请确保数据输入完整平衡。错误详情: {e}")

# ================= 10. 过程能力分析 (Cp/Cpk) =================
elif analysis_type == "10. 过程能力分析 (Cp/Cpk)":
    st.header("过程能力分析 (Cp/Cpk)")
    st.markdown("""
    > 🎯 **应用场景**：对比多组工艺参数，评估哪一组**更稳定**（波动小）且**更准**（趋近目标值）。
    > 📌 **判断标准**：
    > * **Cp (稳定性)**：仅衡量分布宽窄。$> 1.33$ 代表工艺稳定。
    > * **Cpk (综合水平)**：衡量数据是否集中且对准了靶心。**Cpk 越高越好，理想状态 $\ge 1.33$**。
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        usl = st.number_input("规格上限 (USL):", value=10.5, step=0.1)
    with col2:
        lsl = st.number_input("规格下限 (LSL):", value=9.5, step=0.1)
        
    num_groups = st.number_input("请选择要对比的工艺组数 (1 到 5 组):", min_value=1, max_value=5, value=3, step=1)
    
    cols = [f"工艺 {chr(ord('A') + i)}" for i in range(num_groups)]
    default_data = {}
    for i, col in enumerate(cols):
        if i == 0: default_data[col] = [9.8, 9.9, 10.1, 10.0, 10.2, 9.7, 9.9, 10.1]
        elif i == 1: default_data[col] = [10.0, 10.1, 9.9, 10.0, 10.0, 9.9, 10.1, 10.0]
        elif i == 2: default_data[col] = [10.3, 10.4, 9.5, 9.8, 10.6, 9.7, 10.5, 9.6]
        else: default_data[col] = [np.nan] * 8

    df_default = pd.DataFrame(default_data)
    edited_df = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
    
    if st.button("🚀 运行能力评估"):
        if usl <= lsl:
            st.error("❌ 规格上限 (USL) 必须大于规格下限 (LSL)！")
        else:
            results = []
            valid_data_dict = {}
            
            for col in cols:
                d = pd.to_numeric(edited_df[col], errors='coerce').dropna().tolist()
                if len(d) > 2:
                    valid_data_dict[col] = d
                    mean = np.mean(d)
                    std = np.std(d, ddof=1)
                    
                    cp = (usl - lsl) / (6 * std) if std > 0 else 0
                    cpu = (usl - mean) / (3 * std) if std > 0 else 0
                    cpl = (mean - lsl) / (3 * std) if std > 0 else 0
                    cpk = min(cpu, cpl)
                    
                    results.append({
                        "工艺组别": col,
                        "均值 (Mean)": round(mean, 3),
                        "波动 标准差 (Std)": round(std, 3),
                        "Cp (稳定性)": round(cp, 2),
                        "Cpk (综合水平)": round(cpk, 2)
                    })
            
            if results:
                st.subheader("🏆 评估结果排行榜")
                res_df = pd.DataFrame(results)
                st.dataframe(res_df.style.highlight_max(subset=['Cpk (综合水平)'], color='lightgreen'))
                
                st.markdown("### 📊 分布形态对比")
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = sns.color_palette("Set1", len(valid_data_dict))
                
                for i, (name, data) in enumerate(valid_data_dict.items()):
                    sns.kdeplot(data, ax=ax, label=f"{name} (Cpk={res_df.iloc[i]['Cpk (综合水平)']})", color=colors[i], fill=True, alpha=0.3)
                
                ax.axvline(usl, color='red', linestyle='--', label=f'USL ({usl})')
                ax.axvline(lsl, color='red', linestyle='--', label=f'LSL ({lsl})')
                
                ax.set_title("工艺能力分布曲线")
                ax.legend()
                st.pyplot(fig)
                
                # ================= 🌟 修复后的智能大白话诊断逻辑 🌟 =================
                max_cpk = res_df['Cpk (综合水平)'].max()
                best_group = res_df.loc[res_df['Cpk (综合水平)'].idxmax()]
                
                if max_cpk < 0:
                    st.error(f"🚨 **严重警告：所有工艺均已脱靶！**\n\n目前所有工艺的 Cpk 均小于 0，说明产品均值已完全偏离公差带（图上可见全部在红线外）。\n虽然数学上 {best_group['工艺组别']} 得分最高（{max_cpk}），但**在工程上绝对不能采用它**！\n\n💡 **专家建议**：请观察上方分布图，找出**曲线最窄（Cp 稳定性最高）**的工艺。该工艺的重复性极好，只需调整机器中心参数（Offset）将其向左或向右平移至靶心，即可成为最佳工艺！")
                
                elif max_cpk < 1.0:
                    st.warning(f"⚠️ **勉强及格**：推荐使用 **{best_group['工艺组别']}** 的参数（Cpk={max_cpk}）。但注意，它的 Cpk 仍然小于 1.0，说明有一定的废品率风险，建议继续优化参数压缩波动。")
                    
                else:
                    st.success(f"🏆 **终极建议**：强烈推荐使用 **{best_group['工艺组别']}** 的参数！它的 Cpk 最高（{max_cpk}），说明它不仅波动控制得好（分布窄），而且均值稳稳地打在了规格中心地带。")
                # =====================================================================
                
            else:
                st.warning("每组至少需要输入 3 个有效数据才能计算过程能力！")
