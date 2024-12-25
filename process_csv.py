import os
import pandas as pd

def process_csv_files(input_folder):
    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            input_file = os.path.join(input_folder, file_name)
            
            try:
                # 读取CSV文件，跳过第一行
                df = pd.read_csv(input_file, skiprows=1)
                
                # 删除空白行
                df = df.dropna()
                
                # 重命名列
                df.columns = ['X(min)','Y(Counts)']
                
                # 添加行号列
                df.insert(0, 'Point', range(len(df)))
                
                # 准备新的文件内容
                with open(input_file, 'w', encoding='utf-8') as a:
                    # 写入新的头部
                    a.write(f'#"LC_positive {file_name}"\n')
                    a.write('#Point,X(min),Y(Counts)\n')
                    
                    # 写入数据，确保没有空行
                    output = df.to_csv(index=False, float_format='%.10g', header=False, lineterminator='\n')
                    a.write(output.strip())
                
                print(f"已处理: {file_name}")
            except Exception as e:
                print(f"处理失败: {file_name}, 错误: {e}")

# 输入文件夹路径
input_folder = "data_LC"

# 执行处理
process_csv_files(input_folder)
