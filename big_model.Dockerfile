FROM python:3.8

RUN apt update
RUN pip install --upgrade pip
RUN pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
RUN pip config set install.trusted-host repo.huaweicloud.com

# Required by OpenCV
RUN apt install -y libgl1-mesa-glx

# install mindspore
RUN pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.8.0/MindSpore/cpu/x86_64/mindspore-1.8.0-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# install dependencies of object detection application
RUN pip install mindvision torch scikit-image
RUN pip install xlrd==1.2.0 # read xlsx

# install requirements of sedna lib
COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY ./model /home/work/model
COPY big_model/big_model.py  /home/work/infer.py
COPY big_model/interface.py  /home/work/interface.py

ENTRYPOINT ["python"]
CMD ["infer.py"]