import scipy.stats as stats
import pandas as pd
import os
import numpy as np

# ===================== 统计显著性检验模块 =====================
class StatisticalTest:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def paired_ttest(self, group1, group2, model1_name, model2_name):
        t_stat, p_value = stats.ttest_rel(group1, group2)
        
        result = {
            'test_type': 'Paired t-test',
            'model1': model1_name,
            'model2': model2_name,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'alpha': self.alpha
        }
        
        print(f"\n{model1_name} vs {model2_name} 配对t检验结果:")
        print(f"t统计量: {t_stat:.4f}, p值: {p_value:.4f}")
        if p_value < self.alpha:
            print(f"结论: 在{self.alpha}显著性水平下，两个模型性能存在显著差异")
        else:
            print(f"结论: 在{self.alpha}显著性水平下，两个模型性能无显著差异")
        
        return result
    

    def wilcoxon_test(self, group1, group2, model1_name, model2_name):

        group1 = np.array(group1)
        group2 = np.array(group2)

        # 1️⃣ 删除 NaN
        mask = ~np.isnan(group1) & ~np.isnan(group2)
        group1 = group1[mask]
        group2 = group2[mask]

        # 2️⃣ 样本数量检查
        if len(group1) < 2:
            print(f"\n{model1_name} vs {model2_name} Wilcoxon检验:")
            print("样本数量不足，无法进行统计检验")
            return None

        # 3️⃣ 检查差值
        diff = group1 - group2

        if np.all(diff == 0):
            print(f"\n{model1_name} vs {model2_name} Wilcoxon检验:")
            print("两个模型结果完全相同，无法进行Wilcoxon检验")

            result = {
                'test_type': 'Wilcoxon signed-rank test',
                'model1': model1_name,
                'model2': model2_name,
                'w_statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'alpha': self.alpha
            }

            return result

        # 4️⃣ 正常执行检验
        w_stat, p_value = stats.wilcoxon(group1, group2)

        result = {
            'test_type': 'Wilcoxon signed-rank test',
            'model1': model1_name,
            'model2': model2_name,
            'w_statistic': w_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'alpha': self.alpha
        }

        print(f"\n{model1_name} vs {model2_name} Wilcoxon检验结果:")
        print(f"W统计量: {w_stat:.4f}, p值: {p_value:.4f}")

        if p_value < self.alpha:
            print(f"结论: 在{self.alpha}显著性水平下，两个模型性能存在显著差异")
        else:
            print(f"结论: 在{self.alpha}显著性水平下，两个模型性能无显著差异")

        return result
    
    def summarize_significance(self, all_tests, output_dir):
        summary = {
            'significant_tests': sum(1 for test in all_tests if test['significant']),
            'total_tests': len(all_tests),
            'significant_ratio': sum(1 for test in all_tests if test['significant']) / len(all_tests) if len(all_tests) > 0 else 0
        }
        
        test_results_df = pd.DataFrame(all_tests)
        test_results_df.to_excel(os.path.join(output_dir, 'statistical_test_results.xlsx'), index=False)
        
        print(f"\n=== 统计检验汇总 ===")
        print(f"总检验次数: {summary['total_tests']}")
        print(f"显著差异次数: {summary['significant_tests']}")
        print(f"显著差异比例: {summary['significant_ratio']:.2%}")
        
        return summary