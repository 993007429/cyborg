import os
# folder_path ='D:\\xuleSpace\\cyborg\\cyborg\\modules\\ai\\application'
import os
from collections import defaultdict


# def get_substring_before_last_character(string, character):
#     pattern = f".*?{re.escape(character)}"
#     match = re.search(pattern, string[::-1])
#     if match:
#         substring_reverse = match.group(0)
#         substring = substring_reverse[::-1][1:]
#         return substring
#     return None


def get_filenames(directory):
    # 获取文件夹下的各级文件名
    filenames = defaultdict(list)
    for root, dirs, files in os.walk(directory):
        pre_path = root.split('\\')[-1]
        if pre_path == '__pycache__':
            continue
        for file in files:
            filenames[pre_path].append(file.split('.')[0])
    return filenames


if __name__ == '__main__':
    folder_path = 'D:\\xuleSpace\\cyborg\\cyborg\\modules\\ai'
    a = get_filenames(folder_path)
    for k, v in a.items():
        print(k, v)
import re

#
# def get_substring_before_last_character(string, character):
#     pattern = f".*?{re.escape(character)}"
#     match = re.search(pattern, string[::-1])
#     if match:
#         substring_reverse = match.group(0)
#         substring = substring_reverse[::-1][1:]
#         return substring
#     return None


# print(get_substring_before_last_character('D:\\xuleSpace\\cyborg\\cyborg\\modules\\ai\\application', '\\'))
