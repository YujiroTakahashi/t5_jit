FROM pytorch/pytorch:latest AS main-image

ENV LANG=ja_JP.UTF-8 \
    LANGUAGE=ja
RUN set -xe \
    && apt-get update \
    && apt-get install -y --install-recommends \
        sudo \
    && pip install -U \
        protobuf \
        sentencepiece \
        torchtext \
        transformers \
    && groupadd -g 1000 croco \
    && useradd -m -s /bin/bash -u 1000 -g 1000 croco \
    && echo "croco ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && echo "alias ls='ls --color=auto'" > /root/.bashrc
USER croco
CMD ["bash"]
