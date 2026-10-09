import os
import pandas as pd
import re

def extract_metrics_data(folder_path):
    """
    提取指定文件夹下所有带metrics的Excel文件中的数据，并按模型名称汇总
    
    Args:
        folder_path: 包含metrics文件的文件夹路径
    
    Returns:
        汇总后的DataFrame
    """
    # 存储所有模型的metrics数据
    all_metrics = []
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理包含metrics且后缀为xlsx的文件
        if "metrics" in filename and filename.endswith(".xlsx"):
            # 提取模型名称（去掉_metrics.xlsx部分）
            model_name = filename.replace("_metrics.xlsx", "")
            
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 读取Excel文件
                df = pd.read_excel(file_path)
                
                # 确保数据格式正确
                if not df.empty:
                    # 获取第一行数据（实验结果）
                    metrics_data = df.iloc[0].to_dict()
                    
                    # 添加模型名称
                    metrics_data["model"] = model_name
                    
                    # 添加到列表中
                    all_metrics.append(metrics_data)
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
    
    # 将列表转换为DataFrame
    if all_metrics:
        result_df = pd.DataFrame(all_metrics)
        
        # 重新排列列顺序，将model列放在第一列
        cols = ["model"] + [col for col in result_df.columns if col != "model"]
        result_df = result_df[cols]
        
        return result_df
    else:
        return pd.DataFrame()

def main():
    # 设置文件夹路径（请根据实际情况修改）
    folder_path = "./out"  # 当前目录下的out文件夹
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在！")
        return
    
    # 提取数据
    print("正在提取metrics数据...")
    metrics_df = extract_metrics_data(folder_path)
    
    if not metrics_df.empty:
        # 保存汇总结果到Excel文件
        output_file = "./analysis/all_models_metrics_summary.xlsx"
        metrics_df.to_excel(output_file, index=False)
        
        print(f"\n数据提取完成！共找到 {len(metrics_df)} 个模型的metrics数据")
        print(f"汇总结果已保存到: {output_file}")
        
        # 打印预览
        print("\n数据预览：")
        print(metrics_df)
    else:
        print("未找到任何有效的metrics数据！")

if __name__ == "__main__":
    main()