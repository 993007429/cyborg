FROM registry.cn-hangzhou.aliyuncs.com/dipath/cyborg:base

WORKDIR /data/www/cyborg

ADD ./cyborg ./cyborg
ADD ./app.py ./app.py
ADD ./encrypt.py ./encrypt.py
ADD ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

RUN mkdir -p /data/model
RUN mkdir -p /data/download
ADD /data/model/AI /data/model/AI

RUN python encrypt.py build_ext --inplace