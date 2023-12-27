import sys

from Cython.Build import cythonize
import os
import shutil
from setuptools import Extension, setup
from shutil import rmtree

"""
1.安装cython
2.选择需要加密的目录
3.运行python encrypt.py build_ext --inplace

注意这样编译会和使用numba的jit造成冲突

"""


def walk_dir(data_dir, file_types=['.py']):
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:
                if f.lower().endswith(this_type):
                    path_list.append(os.path.join(dirpath, f))
                    break
    return path_list


if __name__ == '__main__':
    compiled_ext = '.so'

    ignore_list = []
    compile_ext = ['.py']

    build_dir = os.getcwd()
    algo_dir = build_dir + '/cyborg/modules/ai/libs/algorithms'

    path_list = os.listdir(algo_dir)

    algo_whitelist = ['Her2_v1']

    for filepath in path_list:
        if filepath not in algo_whitelist:
            shutil.rmtree(os.path.join(algo_dir, filepath), ignore_errors=True)
            continue

        if filepath not in ignore_list:
            file_list = []
            if os.path.isdir(os.path.join(algo_dir, filepath)):
                file_list = walk_dir(os.path.join(algo_dir, filepath), compile_ext)
            else:
                if os.path.splitext(filepath)[1] in compile_ext:
                    file_list = [os.path.join(algo_dir, filepath)]
            for f in file_list:
                if os.path.basename(f).startswith("__init__") or os.path.basename(f) in ignore_list:
                    continue
                if 'Her2New_' in f:
                    if 'utils' in f:
                        continue
                    if 'common.py' in f:
                        continue
                    if 'main.py' in f:
                        continue
                elif 'Her2_v1' in f:
                    if 'utils' in f:
                        continue
                    if 'common.py' in f:
                        continue
                    if 'cell_process_main.py' in f:
                        continue
                    if 'region_process_main.py' in f:
                        continue

                extensions = [Extension(os.path.splitext(os.path.basename(f))[0], [f])]
                setup(ext_modules=cythonize(extensions, compiler_directives={'always_allow_keywords':True}))

                compiled_lib_list = walk_dir(build_dir, [compiled_ext])
                for lib_file in compiled_lib_list:
                    if os.path.basename(lib_file).startswith(os.path.splitext(os.path.basename(f))[0]):
                        shutil.move(lib_file, os.path.join(os.path.dirname(f), os.path.basename(lib_file)))
                        os.remove(f)
                        if os.path.exists(os.path.splitext(f)[0]+'.c'):
                            os.remove(os.path.splitext(f)[0]+'.c')
                        break
    rmtree(os.path.join(build_dir, 'build'), ignore_errors=True)
    print("代码编译完成")
