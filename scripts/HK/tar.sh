#!/bin/bash
# ====================================================================
# 并行批量解压指定目录下所有非 .partial 的 tar 压缩包
# 已存在的文件将直接跳过，支持进度显示
# ====================================================================

SOURCE_DIR="/home/projects/protein/lutianyu/data/AFDB/cityu-data/tsinghua-ai4science/part_00"
DEST_DIR="/home/projects/protein/lutianyu/data/AFDB/AF_part00"
JOBS=200  # ⚙️ 并行任务数，可根据 CPU 核数调整

mkdir -p "$DEST_DIR"

echo "========================================="
echo "开始并行解压 AlphaFold 数据"
echo "源目录: $SOURCE_DIR"
echo "目标目录: $DEST_DIR"
echo "并行任务数: $JOBS"
echo "-----------------------------------------"

# 统计 tar 文件总数
TOTAL=$(find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.tar" ! -name "*.partial*" | wc -l)
echo "总共待处理 $TOTAL 个 tar 包"

# ✅ 用 GNU parallel 并行执行解压任务
find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.tar" ! -name "*.partial*" \
| parallel --bar -j "$JOBS" '
    FNAME=$(basename {})
    echo "--> [开始] 解压 $FNAME"
    tar -xf {} -C '"$DEST_DIR"' --skip-old-files
    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[完成] $FNAME ✅"
    else
        echo "[失败] $FNAME ❌"
    fi
'

echo "-----------------------------------------"
echo "所有任务已完成 ✅"
echo "========================================="