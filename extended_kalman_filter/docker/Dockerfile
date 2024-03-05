FROM osrf/ros:noetic-desktop

WORKDIR /

RUN apt -y update \
    && apt install -y cmake \
    && apt install -y --no-install-recommends build-essential \
    && apt install -y vim \
    && echo "export ROS_HOSTNAME=localhost" >> /root/.bashrc \
    && echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

RUN apt install -y python3-pip

RUN apt -y update \
	&& pip install scipy \
	&& pip install sympy 

COPY ros_ws /ros_ws

WORKDIR /ros_ws
