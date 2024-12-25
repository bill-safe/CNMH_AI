import os
import pandas as pd

def convert_xlsx_to_csv(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.xlsx'):  # 只处理 .xlsx 文件
            input_file = os.path.join(input_folder, file_name)
            # Remove 'negative' from filename and change extension to .csv
            new_filename = file_name.replace(' positive.xlsx', '.csv')
            output_file = os.path.join(output_folder, new_filename)
            
            try:
                # 使用 pandas 读取 Excel 文件
                df = pd.read_excel(input_file, engine='openpyxl')
                # 保存为 CSV 文件
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"已转换: {file_name} -> {output_file}")
            except Exception as e:
                print(f"转换失败: {file_name}, 错误: {e}")

# 输入和输出文件夹路径
input_folder = "LC-MS positive"  # 替换为实际的输入文件夹路径
output_folder = "data_LC"  # 替换为实际的输出文件夹路径

convert_xlsx_to_csv(input_folder, output_folder)
