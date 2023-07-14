FROM dipath/cyborg:0.0.1

WORKDIR /data/www/cyborg/
ADD ./cyborg ./cyborg
ADD ./app.py ./app.py
ADD ./requirements.txt ./requirements.txt
RUN mkdir -p ./local_settings \
    && mkdir -p /data/logs \
    && pip3 install -r requirements.txt

RUN groupadd --gid 10000 dipath \
    && useradd --home-dir /home/dipath --create-home --uid 10000 --gid 10000 --shell /bin/sh --skel /dev/null www \
    && chown dipath:dipath -R . \
    && chmod 555 -R ./ \
    && chmod 777 -R /data/logs

USER dipath