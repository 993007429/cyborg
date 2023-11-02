FROM registry.cn-hangzhou.aliyuncs.com/dipath/cyborg:base

WORKDIR /data/www/cyborg

ADD ./cyborg ./cyborg
ADD ./app.py ./app.py
ADD ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt