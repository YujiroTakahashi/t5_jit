# T5モデルのtorch jit trace

## モデルの取得

```
cd /opt
mkdir model

git lfs install
git clone https://huggingface.co/sonoisa/t5-qiita-title-generation
```

[Hugging Face：sonoisa/t5-qiita-title-generation](https://huggingface.co/sonoisa/t5-qiita-title-generation)


## toch jit traceしたモデルの作成

```
docker-compose up -d
docker-compose exec shell ./build.py

ls opt/
build.py  title.py  traced_jit.pt
```

## toch script のロード

```
docker-compose up -d
docker-compose exec shell ./title.py

title: インターネットで"誰か"を幸せにする
```