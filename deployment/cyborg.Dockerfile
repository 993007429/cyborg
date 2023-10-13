FROM dipath/cyborg:4.0.0

WORKDIR /data/www/cyborg

ADD ./cyborg ./cyborg
ADD ./app.py ./app.py
ADD ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
