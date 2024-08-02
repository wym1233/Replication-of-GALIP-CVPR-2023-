FROM 10.11.3.8:5000/bitahub/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN pip install torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install easydict -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install pandas==1.3.4 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install numpy==1.19.5 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install scipy==1.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install ftfy -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install regex -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install packaging -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install git+https://gitee.com/mirror-sd/CLIP.git && \
    pip cache purge