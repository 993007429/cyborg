FROM dipath/cyborg:4.0.0

WORKDIR /data/www/cyborg

ADD ./cyborg ./cyborg
ADD ./app.py ./app.py
ADD ./encrypt.py ./encrypt.py
ADD ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

RUN mkdir -p /data/model/
ADD /data/model/AI /data/model/AI

RUN python encrypt.py build_ext --inplace