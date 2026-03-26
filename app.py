import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 页面配置
st.set_page_config(page_title="在线质量统计分析中心", layout="wide")
st.title("📊 Web 版 Minitab - 质量统计分析工具")
st.markdown("用于快速执行假设检验与方差分析。支持直接从 Excel 复制粘贴数据。")

# 侧边栏选择分析类型
analysis_type = st.sidebar.radio(
    "请选择统计分析工具：",
    ("单样本 T检验", "双样本 T检验", "单因素方差分析 (ANOVA)")
)

# 辅助函数：解析文本框输入的数据
def parse_data(input_string):
    if not input_string.strip():
        return []
    # 支持空格、逗号、换行符分隔的数据
    raw_data = re.split(r'[\s,]+', input_string.strip())
    try:
        return [float(x) for x in raw_data if x]
    except ValueError:
        st.error("数据解析错误！请确保输入的全是数字。")
        return []

# ---------------- 1. 单样本 T检验 ----------------
if analysis_type == "单样本 T检验":
    st.header("单样本 T检验 (1-Sample t-test)")
    st.markdown("**应用场景**：检验一批产品的平均值是否达到了设计目标（例如：电机转速是否等于 3000 RPM）。")
    
    col1, col2 = st.columns(2)
    with col1:
        target_mean = st.number_input("输入目标均值 (Target Mean):", value=3000.0)
        data_input = st.text_area("在此粘贴测试数据 (空格或回车分隔):", 
                                  "2985 2990 3005 2980 2995 3010 2990 3000 2985 2995")
    
    if st.button("开始分析 🚀"):
        data = parse_data(data_input)
        if len(data) < 2:
            st.warning("请至少输入2个数据点。")
        else:
            # 计算 T检验
            t_stat, p_value = stats.ttest_1samp(data, target_mean)
            mean_val = np.mean(data)
            
            st.subheader("💡 分析结果")
            st.write(f"- **样本均值**: {mean_val:.2f}")
            st.write(f"- **T 统计量**: {t_stat:.4f}")
            st.write(f"- **P 值 (P-value)**: {p_value:.4f}")
            
            if p_value < 0.05:
                st.error(f"**结论**：P 值 < 0.05，拒绝原假设。样本均值与目标值 {target_mean} 存在**显著差异**。")
            else:
                st.success(f"**结论**：P 值 >= 0.05，接受原假设。样本均值与目标值 {target_mean} **无显著差异**（属正常波动）。")

# ---------------- 2. 双样本 T检验 ----------------
elif analysis_type == "双样本 T检验":
    st.header("双样本 T检验 (2-Sample t-test)")
    st.markdown("**应用场景**：比较两批独立产品的平均值是否有显著差异（例如：新旧工艺拉拔力对比，或A、B两台压铸机打出的孔隙率对比）。")
    
    col1, col2 = st.columns(2)
    with col1:
        data_a_input = st.text_area("组别 A 数据 (如：旧工艺):", "15.2 14.8 15.5 14.9 15.1 15.3 14.7")
    with col2:
        data_b_input = st.text_area("组别 B 数据 (如：新工艺):", "16.1 15.9 16.5 16.2 16.0 16.3 16.1")
        
    if st.button("开始分析 🚀"):
        data_a = parse_data(data_a_input)
        data_b = parse_data(data_b_input)
        
        if len(data_a) < 2 or len(data_b) < 2:
            st.warning("每组请至少输入2个数据点。")
        else:
            # 执行方差齐性检验 (Levene's Test)
            stat, p_levene = stats.levene(data_a, data_b)
            equal_var = True if p_levene > 0.05 else False
            
            # 计算 T检验
            t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=equal_var)
            
            st.subheader("💡 分析结果")
            st.write(f"- **组别A 均值**: {np.mean(data_a):.2f} | **组别B 均值**: {np.mean(data_b):.2f}")
            st.write(f"- **方差齐性 (P值)**: {p_levene:.4f} (方差{'相等' if equal_var else '不相等'})")
            st.write(f"- **P 值 (P-value)**: {p_value:.4f}")
            
            if p_value < 0.05:
                st.error("**结论**：P 值 < 0.05，两组数据存在**显著差异**！")
            else:
                st.success("**结论**：P 值 >= 0.05，两组数据**无显著差异**！")
                
            # 画个箱线图
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=[data_a, data_b], ax=ax, palette="Set2")
            ax.set_xticklabels(['Group A', 'Group B'])
            ax.set_title("Boxplot of Group A vs Group B")
            st.pyplot(fig)

# ---------------- 3. 单因素方差分析 (ANOVA) + Tukey ----------------
elif analysis_type == "单因素方差分析 (ANOVA)":
    st.header("单因素方差分析 (One-Way ANOVA) & Tukey 事后检验")
    st.markdown("**应用场景**：比较 3 个或以上组别的均值差异，并找出具体是谁不一样（例如：对比 3 家表面处理供应商的防腐时间）。")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        group1 = st.text_area("供应商 A 数据:", "45 42 48 46 44")
    with col2:
        group2 = st.text_area("供应商 B 数据:", "43 40 45 41 42")
    with col3:
        group3 = st.text_area("供应商 C 数据:", "55 52 58 56 54") # 故意造一组偏高的数据

    if st.button("开始分析 🚀"):
        d1 = parse_data(group1)
        d2 = parse_data(group2)
        d3 = parse_data(group3)
        
        if len(d1) < 2 or len(d2) < 2 or len(d3) < 2:
            st.warning("每组请至少输入2个数据点。")
        else:
            # 1. 执行 ANOVA
            f_stat, p_value = stats.f_oneway(d1, d2, d3)
            
            st.subheader("💡 1. ANOVA 整体分析结果")
            st.write(f"- **F 统计量**: {f_stat:.4f}")
            st.write(f"- **P 值 (P-value)**: {p_value:.4f}")
            
            if p_value >= 0.05:
                st.success("**结论**：P 值 >= 0.05，这三组数据之间没有统计学上的显著差异。无需进行 Tukey 检验。")
            else:
                st.error("**结论**：P 值 < 0.05，说明这三组里**至少有一组**与众不同！接下来自动执行 Tukey 检验抓出“真凶”。")
                
                # 2. 执行 Tukey 事后检验
                st.subheader("💡 2. Tukey 事后检验 (找茬环节)")
                
                # 整合数据以供 statsmodels 使用
                all_values = np.concatenate([d1, d2, d3])
                labels = ['A'] * len(d1) + ['B'] * len(d2) + ['C'] * len(d3)
                
                tukey_results = pairwise_tukeyhsd(endog=all_values, groups=labels, alpha=0.05)
                
                # 打印 Tukey 的表格结果
                st.text(tukey_results.summary())
                st.markdown("""
                **Tukey 表格怎么看？**
                看最后一列 `reject`（是否拒绝原假设）。如果为 **True**，说明这两组之间有**显著差异**；如果为 **False**，说明这两组没区别。
                """)
                
                # 箱线图可视化
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=[d1, d2, d3], ax=ax, palette="Set1")
                ax.set_xticklabels(['Group A', 'Group B', 'Group C'])
                ax.set_title("ANOVA Boxplot")
                st.pyplot(fig)
