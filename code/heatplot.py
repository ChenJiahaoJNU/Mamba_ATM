import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ======================== Basic Settings ========================
# English font only, no Chinese dependency
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
# Fix plot layout warning
plt.rcParams['figure.constrained_layout.use'] = True

# ======================== Data Reading ========================
def read_excel_data(file_path, sheet_name=0):
    """
    Read model pairwise test data from Excel
    :param file_path: Excel file path
    :param sheet_name: Sheet name/index
    :return: DataFrame, all models list, model-to-index map
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Get all unique models and sort (fixed order for symmetry)
    all_models = sorted(list(set(df['模型A'].unique()) | set(df['模型B'].unique())))
    model_to_idx = {model: i for i, model in enumerate(all_models)}
    
    print(f"✅ Read data: {len(df)} records, {len(all_models)} models")
    return df, all_models, model_to_idx

# ======================== Symmetric Matrix Construction (100% Guarantee) ========================
def create_100_symmetric_matrix(df, models, model_to_idx, value_col):
    """
    Create 100% strict symmetric matrix (data layer)
    :param df: Original data
    :param models: Sorted model list (fixed order)
    :param model_to_idx: Model-to-index map
    :param value_col: Target column (e.g., 't检验p值')
    :return: Strict symmetric matrix
    """
    n = len(models)
    matrix = np.ones((n, n))  # Initialize with 1.0 (non-significant) for all
    
    # Fill matrix with bidirectional assignment (core for symmetry)
    for _, row in df.iterrows():
        a, b = row['模型A'], row['模型B']
        val = row[value_col]
        i, j = model_to_idx[a], model_to_idx[b]
        # Force symmetric assignment (overwrite to ensure consistency)
        matrix[i, j] = val
        matrix[j, i] = val
    
    # Diagonal: self-comparison, set to 1.0 (no statistical meaning)
    np.fill_diagonal(matrix, 1.0)
    
    # Strict symmetry check (raise error if not symmetric)
    assert np.allclose(matrix, matrix.T, atol=1e-10), "❌ Matrix is not symmetric!"
    print(f"✅ {value_col} - Strict symmetric matrix created (size: {n}x{n})")
    return matrix

# ======================== Heatmap Plot (Separate Figures) ========================
def plot_separate_heatmaps(t_matrix, w_matrix, models, save_base_path):
    """
    Plot t-test and Wilcoxon test heatmaps as SEPARATE figures
    :param t_matrix: t-test symmetric matrix
    :param w_matrix: Wilcoxon test symmetric matrix
    :param models: Sorted model list
    :param save_base_path: Base path for saving (e.g., './analysis/model_heatmap_')
    """
    n = len(models)
    # Professional color map: Red(Significant) → White → Blue(Non-significant)
    colors = ['#D73027', '#F46D43', '#FEE090', '#E0F3F8', '#ABD9E9', '#74ADD1', '#4575B4']
    cmap = LinearSegmentedColormap.from_list('sig_cmap', colors, N=256)
    
    # ---------------- Common plot function (unify style) ----------------
    def plot_single_heatmap(matrix, title, save_path):
        # Create independent figure with EQUAL aspect ratio
        fig, ax = plt.subplots(1, 1, figsize=(16, 14), constrained_layout=True)
        fig.suptitle(f'Model Pairwise Comparison Test - {title}', 
                     fontsize=20, fontweight='bold', y=1.02)
        
        # Plot with EQUAL aspect (100% visual symmetry)
        im = ax.imshow(matrix, cmap=cmap, aspect='equal', vmin=0, vmax=0.05)
        
        # Set ticks (unify font size/rotation)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12, fontweight='medium')
        ax.set_yticklabels(models, fontsize=12, fontweight='medium')
        
        # Add white grid line (separate cells)
        for i in range(n):
            ax.axhline(i-0.5, color='white', linewidth=0.4)
            ax.axvline(i-0.5, color='white', linewidth=0.4)
        
        # Color bar
        cbar = plt.colorbar(im, ax=ax, shrink=0.9, aspect=40, pad=0.03)
        cbar.set_label('p-value', fontsize=14, fontweight='bold')
        cbar.set_ticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
        cbar.set_ticklabels(['0.00', '0.01', '0.02', '0.03', '0.04', '0.05'], fontsize=12)
        
        # Add subtitle explanation
        ax.text(0.5, -0.1, 'Red: p<0.05 (Significant) | Blue: p≥0.05 (Non-significant)', 
                transform=ax.transAxes, ha='center', fontsize=12, fontweight='medium')
        
        # Save high-resolution heatmap
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"✅ Heatmap saved to: {save_path}")
    
    # ---------------- Plot and save separate heatmaps ----------------
    # Save t-test heatmap
    t_save_path = f"{save_base_path}t_test.png"
    plot_single_heatmap(t_matrix, 't-test p-value Heatmap', t_save_path)
    
    # Save Wilcoxon test heatmap
    w_save_path = f"{save_base_path}wilcoxon_test.png"
    plot_single_heatmap(w_matrix, 'Wilcoxon Test p-value Heatmap', w_save_path)

# ======================== Main Function ========================
def main():
    # ---------------- Modify only these 3 parameters ----------------
    EXCEL_PATH = './out/all_models_pairwise_tests_results.xlsx'
    SHEET_NAME = '全量两两检验'  # Only for reading, no display in plot
    SAVE_BASE_PATH = './analysis/model_pairwise_heatmap_'  # Base path for separate files
    # -----------------------------------------------------------------
    
    # Step 1: Read data
    df, all_models, model_to_idx = read_excel_data(EXCEL_PATH, SHEET_NAME)
    # Step 2: Create 100% symmetric matrix (data layer)
    t_test_mat = create_100_symmetric_matrix(df, all_models, model_to_idx, 't检验p值')
    wilcoxon_mat = create_100_symmetric_matrix(df, all_models, model_to_idx, 'Wilcoxon p值')
    # Step 3: Plot and save separate heatmaps
    plot_separate_heatmaps(t_test_mat, wilcoxon_mat, all_models, SAVE_BASE_PATH)
    print("\n🎉 All done! Separate symmetric heatmaps generated successfully.")

# ======================== Run Code ========================
if __name__ == '__main__':
    main()