FROM python:3.8

RUN apt update
RUN pip install --upgrade pip

# Required by OpenCV
RUN apt install -y libgl1-mesa-glx

# install mindspore
RUN pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.8.0/MindSpore/cpu/aarch64/mindspore-1.8.0-cp38-cp38-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
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
COPY joint_inference/helmet_detection_inference/frames /home/work/images
COPY joint_inference/helmet_detection_inference/little_model/little_model.py  /home/work/infer.py
COPY joint_inference/helmet_detection_inference/little_model/interface.py  /home/work/interface.py

#ENTRYPOINT ["python"]
#CMD ["infer.py"]
