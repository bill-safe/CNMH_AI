import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

def plot_filter_comparison(file_paths):
    """
    对多个文件进行Savitzky-Golay滤波对比可视化
    
    Args:
        file_paths: 文件路径列表
    """
    # 创建子图
    n_files = len(file_paths)
    fig, axes = plt.subplots(n_files, 1, figsize=(12, 6*n_files))
    if n_files == 1:
        axes = [axes]
    
    for ax, file_path in zip(axes, file_paths):
        try:
            # 读取数据
            df = pd.read_csv(file_path, skiprows=1)
            raw_data = df['Y(Counts)'].values
            
            # 应用Savitzky-Golay滤波
            filtered_data = savgol_filter(raw_data, window_length=9, polyorder=5)
            
            # 创建x轴数据点
            x = range(len(raw_data))
            
            # 绘制原始数据和滤波后的数据
            ax.plot(x, raw_data, 'b-', alpha=0.5, label='Raw Data')
            ax.plot(x, filtered_data, 'r-', label='Filtered Data')
            
            # 设置标题和标签
            ax.set_title(f'File: {os.path.basename(file_path)}')
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Counts')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # 添加总标题
    plt.suptitle('Savitzky-Golay Filter Comparison\nWindow Length=9, Polyorder=5', 
                 fontsize=16, y=1.02)
    
    # 调整布局并保存
    plt.tight_layout()
    
    # 保存到visualization_results目录
    output_path = os.path.join('visualization_results', 'filter_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 创建输出目录
    if not os.path.exists('visualization_results'):
        os.makedirs('visualization_results')
    
    # 选择多个示例文件进行对比
    example_files = [
        'data_LC_positive/APCK1.csv',  
        'data_LC_positive/FRCK1.csv',  
        'data_LC_positive/RACK1.csv',  
        'data_LC_positive/RHCK1.csv'   
    ]
    
    # 进行可视化对比
    plot_filter_comparison(example_files)
    print("Visualization completed. Results saved in 'visualization_results/filter_comparison.png'")

if __name__ == "__main__":
    # 设置matplotlib样式
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    
    main()
