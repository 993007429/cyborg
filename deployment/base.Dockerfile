FROM registry.cn-shanghai.aliyuncs.com/space1/compile-python:3.8.12

WORKDIR /data/www/sp-saas/
ADD ./requirements.txt ./requirements.txt
ADD ./requirements-dynamic.txt ./requirements-dynamic.txt
RUN pip3 install -i http://mirrors.idiaoyan.cn/repository/pypi/simple/ --trusted-host mirrors.idiaoyan.cn -r requirements.txt
RUN pip3 install -i http://mirrors.idiaoyan.cn/repository/pypi/simple/ --trusted-host mirrors.idiaoyan.cn -r requirements-dynamic.txt
