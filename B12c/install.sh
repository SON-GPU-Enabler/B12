#!/bin/bash

script_dir=$(dirname $0)
cd $script_dir

mkdir -p build

cd build

script_dir=$(cd ../ && pwd)
script_dir=$(basename $script_dir)

if [ ! -f Makefile ]
then
#   CC=gcc-4.9 CXX=g++-4.9 cmake ..
  CC=gcc-9 CXX=g++-9 cmake ..
fi

make -j4
make -j4
make -j4

sudo make install
sudo make install

echo "export PATH=/opt/B12/${script_dir}/"':${PATH}' >> ~/.bashrc
