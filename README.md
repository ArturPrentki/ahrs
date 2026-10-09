# Wymagania
docker, docker compose

# Budowanie

docker compose build 
za pierwszym razem zajmie to około 10-15 minut
to polecenie stworzy 2 obrazy: 
    gardo-brains - kontener, zawierający wszystko, co potrzebne aby uruchomić symulację robota z WBC

    isaac-lab - kontener zawierający isaaclab i skrypty do trenowania sieci neuronowej

# Uruchomienie kontenerów
    docker compose up -d

# Trenowanie sieci
    docker exec -it gardo-isaaclab ./isaaclab.sh -p /gardo_ws/rl_src/hierarchical_wbc/train.py --headless

# Eksportowanie sieci
    TODO

# Budowanie środowiska ROS2
    Uruchomienie docker-a
        docker compose run --rm brains /bin/bash
    w kontenerze
        colcon build

    uruchomienie symulacji robota w gazebo: (host musi udostępnić xhost-a: xhost +local:docker)
        source install/setup.sh
        ros2 launch go2_description launch.py 
