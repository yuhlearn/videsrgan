# videsrgan

[![codecov](https://codecov.io/gh/yuhlearn/videsrgan/branch/main/graph/badge.svg?token=videsrgan_token_here)](https://codecov.io/gh/yuhlearn/videsrgan)
[![CI](https://github.com/yuhlearn/videsrgan/actions/workflows/main.yml/badge.svg)](https://github.com/yuhlearn/videsrgan/actions/workflows/main.yml)

This is a project under development, not everything is in place right now.

## Building on Linux

In order to build the project, you first need the essential Vulkan build requirements

```bash
apt-get install libvulkan-dev
```
```bash
dnf install vulkan-headers vulkan-loader-devel
```
```bash
pacman -S vulkan-headers vulkan-icd-loader
```

Clone the repo and initialize the submodules
```bash
git clone https://github.com/yuhlearn/videsrgan.git
cd videsrgan
git submodule update --init --recursive
```

Build with CMake
```bash
cd videsrgan
mkdir build
cd build
cmake ../src
cmake --build . -j 4
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

```bash
python -m videsrgan
#or
videsrgan
```
