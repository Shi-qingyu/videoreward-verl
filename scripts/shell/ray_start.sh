ray stop

# 选择 eth4 对应的 IP
HEAD_IP=22.12.0.118

# 固定 GCS 端口（默认 6379），并把 Dashboard 绑到 0.0.0.0 方便远程看
ray start --head \
  --node-ip-address "$HEAD_IP" \
  --port 6379 \
  --dashboard-host 0.0.0.0 --dashboard-port 8265 \
  --min-worker-port 30000 --max-worker-port 30100