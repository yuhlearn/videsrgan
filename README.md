# videsrgan

[![codecov](https://codecov.io/gh/yuhlearn/videsrgan/branch/main/graph/badge.svg?token=videsrgan_token_here)](https://codecov.io/gh/yuhlearn/videsrgan)
[![CI](https://github.com/yuhlearn/videsrgan/actions/workflows/main.yml/badge.svg)](https://github.com/yuhlearn/videsrgan/actions/workflows/main.yml)

Awesome videsrgan created by yuhlearn

## Building on Debian/Ubuntu

To build the project you first need the following packages installed

```
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libomp-dev libvulkan-dev vulkan-tools libopencv-dev mesa-vulkan-drivers
```

## Creating a virtualenv

```bash
make virtualenv
source .venv/bin/activate
```

## Install it from PyPI

```bash
pip install videsrgan
```

## Usage

```py
from videsrgan import BaseClass
from videsrgan import base_function

BaseClass().base_method()
base_function()
```

```bash
$ python -m videsrgan
#or
$ videsrgan
```

## Development

