FROM dipath/cyborg:0.0.2

ADD ./cyborg ./cyborg
ADD ./app.py ./app.py
ADD ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txmt
