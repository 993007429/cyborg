FROM dipath/cyborg:4.0.0

WORKDIR /data/www/cyborg

RUN pip install torch==1.13.1 -f https://download.pytorch.org/whl/torch_stable.html