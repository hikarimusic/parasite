# Setup

This is an instruction on how to set up the environment for GPU acceleration on a computer with NVIDIA graphics cards. 
Ubuntu is the recommended operating system. 

## Install NVIDIA driver 

List available drivers.

```sh
ubuntu-drivers devices
```

Install a driver. You should choose the one recommended by your computer.

```sh
sudo apt install nvidia-driver-XXX
```

After having installed the driver, reboot your computer. There might be some password setting about secure boot.

```sh
reboot
```

You can test whether the driver is successfully installed.

```sh
nvidia-smi
```

## Build virtual environment

It is recommended to run the project in an isolated virtual environment for package management.

```sh
python3 -m venv .parasight
```

Remember to activate this virtual environment every time you run the project.

```sh
source .parasight/bin/activate
```

## Clone the repository

```sh
git clone https://github.com/hikarimusic/PARASIGHT.git
```

```sh
cd PARASIGHT
```

Install the required python packages by pip.

```sh
pip install -r requirements.txt
```

Install tkinter as GUI backend of matplotlib.

```sh
sudo apt-get install python3-tk
```

## Check 

You can check whether GPU acceleration is available after setting up the environment.

```sh
>>> import torch
>>> torch.cuda.is_available()
True
```

---

## Cleanup CPU memory

Find the process consuming memory:

```sh
nvidia-smi
```

Kill the process by PID:

```sh
sudo kill -9 PID
```
