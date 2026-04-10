#!/bin/bash

# 检查参数
if [ -z "$1" ]; then
  echo "❌ 请提供一个 commit ID。"
  echo "用法: ./update-submodule.sh <commit-id>"
  exit 1
fi

COMMIT_ID=$1
SUBMODULE_PATH="verl"

# 确保 submodule 目录存在
if [ ! -d "$SUBMODULE_PATH" ]; then
  echo "❌ 子模块目录 '$SUBMODULE_PATH' 不存在。"
  exit 1
fi

echo "➡️ 正在将子模块 '$SUBMODULE_PATH' 回退到 commit: $COMMIT_ID"

# 进入子模块
cd "$SUBMODULE_PATH" || exit 1

# 检出指定 commit
git fetch
git checkout "$COMMIT_ID"

# 返回主项目
cd ..

# 添加并提交变更
git add "$SUBMODULE_PATH"
git commit -m "Update submodule '$SUBMODULE_PATH' to commit $COMMIT_ID"
git push
echo "✅ 子模块已更新并提交。"