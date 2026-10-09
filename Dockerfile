FROM osrf/ros:jazzy-desktop-full

SHELL [ "/bin/bash", "-c" ] 
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y --fix-missing \
    python3-colcon-common-extensions \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-ros-gz \
    ros-jazzy-mujoco-vendor \
    ros-jazzy-mujoco-ros2-control \
    ros-jazzy-mujoco-ros2-control-plugins \
    ros-jazzy-mujoco-ros2-control-demos \
    ros-jazzy-xacro \
    ros-jazzy-osqp-vendor \
    ros-jazzy-pinocchio \
    ros-jazzy-crocoddyl \
    libgl1 \
    libegl1 \
    mesa-utils \
    mesa-utils-extra \
    can-utils \
    clang-format \
    clangd \
    wget \
    tar \
    python3-pip \
    python3-scipy \
    libyaml-cpp-dev \
    libspdlog-dev \
    libboost-all-dev \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-nmea-msgs \
    ros-jazzy-rtcm-msgs \
    && rm -rf /var/lib/apt/lists/*
RUN rosdep update --rosdistro jazzy || true

# source the workspace automatically once it has been built
RUN echo '[ -f /gardo_ws/install/setup.bash ] && source /gardo_ws/install/setup.bash' >> ~/.bashrc

RUN pip install --break-system-packages onnxruntime mujoco

ENV ONNXRUNTIME_VERSION=1.17.1

RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
    tar -xzf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
    cp -r onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/include/* /usr/local/include/ && \
    cp -r onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/* /usr/local/lib/ && \
    rm -rf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}*

RUN git clone https://github.com/unitreerobotics/unitree_sdk2.git /root/unitree_sdk2 && \
    cd /root/unitree_sdk2 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics && \
    make -j$(nproc) && \
    make install

ENV MUJOCO_VERSION=3.3.6

RUN wget -q https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz && \
    mkdir -p /opt/mujoco && \
    tar -xzf mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz -C /opt/mujoco && \
    rm mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz

RUN git clone https://github.com/unitreerobotics/unitree_mujoco.git /root/unitree_mujoco

# patch for gcc 13+
RUN sed -i '1s/^/#include <cstdint>\n/' /root/unitree_mujoco/simulate/src/joystick/joystick.h && \
    sed -i '1s/^/#include <cstdint>\n/' /root/unitree_mujoco/simulate/src/joystick/jstest.cc && \
    mkdir -p /root/unitree_mujoco/simulate/build && \
    mkdir -p /root/unitree_mujoco/simulate/include && \
    ln -s /opt/mujoco/mujoco-${MUJOCO_VERSION} /root/unitree_mujoco/simulate/mujoco
    
RUN cd /root/unitree_mujoco/simulate/build && \
    cmake .. && \
    make -j$(nproc)

ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/unitree_robotics/lib/:$LD_LIBRARY_PATH
RUN ln -s /opt/ros/jazzy/include/mujoco/* /root/unitree_mujoco/simulate/include/ 2>/dev/null || true

WORKDIR /gardo_ws
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
