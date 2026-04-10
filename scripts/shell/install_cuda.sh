#!/bin/bash

cd /opt/tiger
hdfs dfs -get hdfs://harunava/home/byte_ttlive_strategy/lance/pkgs/cuda_12.8.1_570.124.06_linux.run .
sudo sh cuda_12.8.1_570.124.06_linux.run