import os
import shutil
def delete_all_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子文件夹
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
            continue

# 使用示例
folder_path = "C:/Users/yks/Downloads"
delete_all_files_in_folder(folder_path)