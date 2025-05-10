# import os
# import random

# def get_image_paths(root_dir):
#     # 存储所有图片的路径
#     image_paths = []
    
#     # 遍历目录，获取所有图片路径
#     for subdir, _, files in os.walk(root_dir):
#         for file in files:
#             # 判断文件是否为图片，扩展名可以根据需求修改
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#                 image_paths.append(os.path.join(subdir, file))
    
#     return image_paths

# def save_shuffled_paths(image_paths, output_file):
#     # 打乱路径顺序
#     random.shuffle(image_paths)
    
#     # 将打乱后的路径保存到txt文件
#     with open(output_file, 'w') as f:
#         for path in image_paths:
#             f.write(path + '\n')

# # 设置根目录和输出文件路径
# root_dir = '/mnt/ImarsData3/rice_dataset/'
# output_file = '/home/zfj/zzh/yolov5-master/data/水稻数据_0318.txt'

# # 获取所有图片路径
# image_paths = get_image_paths(root_dir)

# # 保存打乱后的路径
# save_shuffled_paths(image_paths, output_file)

# print(f"Shuffled image paths have been saved to {output_file}")
##############################################################################
# import os

# # 定义文件路径
# file1_path = "/home/zfj/zzh/yolov5-master/shuffled_image_paths.txt"
# file2_path = "/home/zfj/zzh/yolov5-master/data/shuffled_image_paths.txt"

# # 读取 file2，提取图片文件名集合
# with open(file2_path, "r") as f:
#     file2_lines = [line.strip() for line in f if line.strip()]
# file2_names = {os.path.basename(path) for path in file2_lines}
# print(file2_names)
# # 读取 file1
# with open(file1_path, "r") as f:
#     file1_lines = [line.strip() for line in f if line.strip()]
# print(file1_lines)

# # 过滤 file1 中的路径，如果其图片名在 file2_names 中，则删除该路径
# filtered_lines = [line for line in file1_lines if os.path.basename(line) not in file2_names]

# # 将过滤后的内容写回 file1（或保存到新文件）
# with open(file1_path, "w") as f:
#     for line in filtered_lines:
#         f.write(line + "\n")

# print("处理完成，结果已保存到：", file1_path)

##############################################################################
# import random

# file_path = '/home/zfj/zzh/yolov5-master/data/rice_data_text/合并水稻数据_test.txt'

# # 读取所有行
# with open(file_path, 'r') as f:
#     lines = f.readlines()

# # 计算需要保留的行数（向下取整）
# num_lines = len(lines)
# num_to_keep = num_lines // 20

# # 随机选择要保留的行
# lines_kept = random.sample(lines, num_to_keep)

# # 将保留的行写回文件
# with open(file_path, 'w') as f:
#     f.writelines(lines_kept)

# print(f"共 {num_lines} 行，随机保留了 {num_to_keep} 行。")

##############################################################################
# import os
# from datetime import datetime

# # 输入文件路径
# input_file = "/home/zfj/zzh/yolov5-master/data/rice_data_text/中山水稻.txt"
# output_file_1 = "/home/zfj/zzh/yolov5-master/data/rice_data_text/中山水稻1.txt"
# output_file_2 = "/home/zfj/zzh/yolov5-master/data/rice_data_text/中山水稻2.txt"

# # 读取数据
# with open(input_file, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# # 解析数据
# data = []
# header = lines[0]  # 读取表头
# for line in lines[1:]:  # 跳过表头
#     parts = line.strip().split("\t")
#     if len(parts) == 2 and parts[1].strip():  # 确保 local_filename 不为空
#         upload_date_str = parts[0].strip().strip('"')  # 去掉两侧的引号
#         try:
#             upload_date = datetime.strptime(upload_date_str, "%d/%m/%Y")  # 解析日期
#             data.append((upload_date, parts[0], parts[1]))  # 存储 (日期, 原始日期字符串, 文件路径)
#         except ValueError:
#             print(f"警告：无法解析日期 {parts[0]}，跳过该行。")

# # 定义日期范围
# start_range_1 = datetime(2024, 2, 20)  # 2024年2月下旬
# end_range_1 = datetime(2024, 7, 1)  # 2024年7月中旬
# start_range_2 = datetime(2024, 7, 1)  # 2024年6月下旬
# end_range_2 = datetime(2024, 11, 10)  # 2024年11月中下旬

# # 按时间过滤数据
# filtered_data_1 = [f"{date_str}\t{filename}\n" for date, date_str, filename in data if start_range_1 <= date <= end_range_1]
# filtered_data_2 = [f"{date_str}\t{filename}\n" for date, date_str, filename in data if start_range_2 <= date <= end_range_2]

# # 写入文件 1
# if filtered_data_1:
#     with open(output_file_1, "w", encoding="utf-8") as f:
#         f.write(header)
#         f.writelines(filtered_data_1)
#     print(f"已生成文件：{output_file_1}")
# else:
#     print("没有符合 2024年2月下旬到2024年7月中旬 条件的数据。")

# # 写入文件 2
# if filtered_data_2:
#     with open(output_file_2, "w", encoding="utf-8") as f:
#         f.write(header)
#         f.writelines(filtered_data_2)
#     print(f"已生成文件：{output_file_2}")
# else:
#     print("没有符合 2024年6月下旬到2024年11月中下旬 条件的数据。")

##############################################################################


# import os
# from datetime import datetime

# # 输入文件路径
# file_1 = "/home/zfj/zzh/yolov5-master/data/rice_data_text/中山水稻2.txt"
# file_2 = "/home/zfj/zzh/yolov5-master/data/rice_data_text/开平水稻2.txt"
# output_file = "/home/zfj/zzh/yolov5-master/data/rice_data_text/合并水稻数据2.txt"

# # 读取数据
# def read_valid_data(file_path):
#     """ 读取文件并返回 (upload_date, 原始日期字符串, 文件路径) 的列表 """
#     with open(file_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     data = []
#     header = lines[0]  # 读取表头
#     for line in lines[1:]:  # 跳过表头
#         parts = line.strip().split("\t")
#         if len(parts) == 2 and parts[1].strip():  # 确保 local_filename 不为空
#             upload_date_str = parts[0].strip().strip('"')  # 去掉引号
#             try:
#                 upload_date = datetime.strptime(upload_date_str, "%d/%m/%Y")  # 解析日期
#                 data.append((upload_date, parts[0], parts[1]))  # 存储 (日期, 原始日期字符串, 文件路径)
#             except ValueError:
#                 print(f"警告：无法解析日期 {parts[0]}，跳过该行。")
    
#     return header, data

# # 读取两个文件的数据
# header_1, data_1 = read_valid_data(file_1)
# header_2, data_2 = read_valid_data(file_2)

# # 确保表头一致，否则取第一个文件的表头
# header = header_1 if header_1 == header_2 else "upload_date\tlocal_filename\n"

# # 合并数据并排序
# merged_data = sorted(data_1 + data_2, key=lambda x: x[0])  # 按日期排序

# # 写入合并后的文件
# with open(output_file, "w", encoding="utf-8") as f:
#     f.write(header)
#     for _, date_str, filename in merged_data:
#         f.write(f"{date_str}\t{filename}\n")

# print(f"合并完成，已生成文件：{output_file}")


##############################################################################



# import os

# # 输入输出文件路径
# input_file = "/home/zfj/zzh/yolov5-master/data/rice_data_text/合并水稻数据_test.txt"
# output_file = "/home/zfj/zzh/yolov5-master/data/rice_data_text/合并水稻数据_test.txt"

# # 读取文件并处理
# with open(input_file, 'r') as file:
#     lines = file.readlines()

# # 处理每一行
# cleaned_lines = []
# for line in lines:
#     # 去掉引号
#     cleaned_line = line.replace('"', '')
    
#     # 只保留包含 '/mnt/ImarsData3' 的行
#     if '/mnt/ImarsData3' in cleaned_line:
#         cleaned_lines.append(cleaned_line)

# # 将处理后的内容写入新文件
# with open(output_file, 'w') as file:
#     file.writelines(cleaned_lines)

# print(f"清理后的文件已保存至：{output_file}")
##############################################################################

# import os
# import shutil
# from datetime import datetime

# # 输入文件路径
# input_file = "/home/zfj/zzh/yolov5-master/data/rice_data_text/合并水稻数据2.txt"
# output_dir = "/home/zfj/zzh/yolov5-master/data/rice_bigdata_2"

# # 读取文件并处理
# with open(input_file, 'r') as file:
#     lines = file.readlines()

# # 跳过表头（第一行）
# lines = lines[1:]

# # 处理每一行
# for line in lines:
#     # 分割每行，得到日期和文件路径
#     parts = line.strip().split("\t")
#     upload_date = parts[0].replace('"', '').strip()  # 处理日期
#     local_filename = parts[1].replace('"', '').strip()  # 处理文件路径

#     # 转换日期格式，假设原日期格式为 'd/m/yyyy'
#     try:
#         formatted_date = datetime.strptime(upload_date, "%d/%m/%Y").strftime("%Y-%m-%d")
#     except ValueError:
#         print(f"日期格式错误: {upload_date}")
#         continue  # 跳过格式错误的日期

#     # 创建日期文件夹路径
#     date_folder_path = os.path.join(output_dir, formatted_date)

#     # 如果文件夹不存在，创建它
#     if not os.path.exists(date_folder_path):
#         os.makedirs(date_folder_path)

#     # 复制图片到对应的日期文件夹
#     if os.path.exists(local_filename):  # 检查文件是否存在
#         shutil.copy(local_filename, os.path.join(date_folder_path, os.path.basename(local_filename)))
#         print(f"已复制 {local_filename} 到 {date_folder_path}")
#     else:
#         print(f"文件不存在: {local_filename}")

# print("所有文件已处理完成。")

# import os
# import cv2

# def delete_corrupt_images(root_dir, report_interval=1500):
#     """删除 root_dir 目录下所有子目录中无法打开的图片，并统计检查和删除的数量，每隔 report_interval 张图片反馈一次"""
#     total_images = 0
#     deleted_images = 0

#     for subdir, _, files in os.walk(root_dir):
#         for file in files:
#             file_path = os.path.join(subdir, file)
#             if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif')):
#                 total_images += 1
#                 img = cv2.imread(file_path)
#                 if img is None:
#                     print(f"删除损坏的图片: {file_path}")
#                     os.remove(file_path)
#                     deleted_images += 1

#                 # 每隔 report_interval 张图片输出一次进度
#                 if total_images % report_interval == 0:
#                     print(f"已检查 {total_images} 张图片，删除 {deleted_images} 张损坏的图片。")

#     print(f"检查完成！总共检查了 {total_images} 张图片，删除了 {deleted_images} 张损坏的图片。")

# root_dir = "/home/zfj/zzh/yolov5-master/data/rice_bigdata_2"
# delete_corrupt_images(root_dir)

