FROM python:3.7

RUN apt update \
  && apt install -y libgl1-mesa-glx
  
COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt
RUN pip install opencv-python==4.5.1.48
RUN pip install Pillow==8.0.1
RUN pip install protobuf==3.20
COPY ./tensorflow-1.15.0-cp37-cp37m-linux_aarch64.whl /home
RUN pip install /home/tensorflow-1.15.0-cp37-cp37m-linux_aarch64.whl

ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

ENTRYPOINT ["python"]
COPY frames /home/work/images
COPY little_model/little_model.py  /home/work/infer.py
COPY little_model/interface.py  /home/work/interface.py

CMD ["infer.py"]  
