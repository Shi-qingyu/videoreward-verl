#! /bin/bash

# 同步verl最新版本
cd verl
git pull origin main
cd ..
git add verl
git commit -m "sync verl"
git push