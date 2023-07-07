FROM cr.idiaoyan.cn/sp/sp-saas:1.16.9

WORKDIR /data/www/sp-saas/
ADD ./seal ./seal
ADD ./tools ./tools
ADD ./app.py ./app.py
ADD ./shell.py ./shell.py
ADD ./requirements.txt ./requirements.txt
ADD ./requirements-dynamic.txt ./requirements-dynamic.txt
RUN echo "Asia/Shanghai" > /etc/timezone \
    && export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
    && mkdir -p ./local_settings \
    && mkdir -p /data/logs \
    && pip3 install -r requirements.txt

RUN groupadd --gid 10000 dipath \
    && useradd --home-dir /home/dipath --create-home --uid 10000 --gid 10000 --shell /bin/sh --skel /dev/null www \
    && chown dipath:dipath -R . \
    && chmod 555 -R ./ \
    && chmod 777 -R /data/logs

USER dipath