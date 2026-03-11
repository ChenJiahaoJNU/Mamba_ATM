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

# ======================== Heatmap Plot (Visual Symmetry Guarantee) ========================
def plot_symmetric_heatmap(t_matrix, w_matrix, models, save_path):
    """
    Plot strictly symmetric heatmap (visual layer)
    :param t_matrix: t-test symmetric matrix
    :param w_matrix: Wilcoxon test symmetric matrix
    :param models: Sorted model list
    :param save_path: Save path for heatmap
    """
    n = len(models)
    # Professional color map: Red(Significant) → White → Blue(Non-significant)
    colors = ['#D73027', '#F46D43', '#FEE090', '#E0F3F8', '#ABD9E9', '#74ADD1', '#4575B4']
    cmap = LinearSegmentedColormap.from_list('sig_cmap', colors, N=256)
    
    # Create figure with EQUAL aspect ratio (key for visual symmetry)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12), constrained_layout=True)
    fig.suptitle('Model Pairwise Comparison Test - p-value Heatmap', 
                 fontsize=24, fontweight='bold', y=1.02)
    
    # ---------------- Common plot function (unify style, avoid visual bias) ----------------
    def plot_single_heatmap(ax, matrix, title):
        # Plot with EQUAL aspect (100% visual symmetry)
        im = ax.imshow(matrix, cmap=cmap, aspect='equal', vmin=0, vmax=0.05)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        
        # Set ticks (unify font size/rotation for x/y)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10, fontweight='medium')
        ax.set_yticklabels(models, fontsize=10, fontweight='medium')
        
        # Add white grid line (separate cells, no visual interference)
        for i in range(n):
            ax.axhline(i-0.5, color='white', linewidth=0.4)
            ax.axvline(i-0.5, color='white', linewidth=0.4)
        
        # Color bar (unify style)
        cbar = plt.colorbar(im, ax=ax, shrink=0.9, aspect=40, pad=0.03)
        cbar.set_label('p-value', fontsize=14, fontweight='bold')
        cbar.set_ticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
        cbar.set_ticklabels(['0.00', '0.01', '0.02', '0.03', '0.04', '0.05'], fontsize=12)
        return im
    
    # ---------------- Plot t-test and Wilcoxon test heatmap ----------------
    plot_single_heatmap(ax1, t_matrix, 't-test\n(Red: p<0.05 Significant | Blue: p≥0.05 Non-significant)')
    plot_single_heatmap(ax2, w_matrix, 'Wilcoxon Test\n(Red: p<0.05 Significant | Blue: p≥0.05 Non-significant)')
    
    # Save high-resolution heatmap (no white border)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Heatmap saved to: {save_path}")

# ======================== Main Function ========================
def main():
    # ---------------- Modify only these 3 parameters ----------------
    EXCEL_PATH = './out/all_models_pairwise_tests_results.xlsx'
    SHEET_NAME = '全量两两检验'  # Only for reading, no display in plot
    SAVE_PATH = './analysis/model_pairwise_heatmap_symmetric.png'
    # -----------------------------------------------------------------
    
    # Step 1: Read data
    df, all_models, model_to_idx = read_excel_data(EXCEL_PATH, SHEET_NAME)
    # Step 2: Create 100% symmetric matrix (data layer)
    t_test_mat = create_100_symmetric_matrix(df, all_models, model_to_idx, 't检验p值')
    wilcoxon_mat = create_100_symmetric_matrix(df, all_models, model_to_idx, 'Wilcoxon p值')
    # Step 3: Plot symmetric heatmap (visual layer)
    plot_symmetric_heatmap(t_test_mat, wilcoxon_mat, all_models, SAVE_PATH)
    print("\n🎉 All done! Strict symmetric heatmap generated successfully.")

# ======================== Run Code ========================
if __name__ == '__main__':
    main()