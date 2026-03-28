#!/bin/bash
# 增强方式对比实验：none / noise / gan × dtn / ftn_u0 / mrn / maml
# 使用方式：bash run_augmentation_eval.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

UV="$HOME/.local/bin/uv"
PYTHON="$UV run --project . python"

METHODS=(dtn ftn_u0 mrn maml)
AUG_TYPES=(none noise gan)
TOTAL=$(( ${#METHODS[@]} * ${#AUG_TYPES[@]} ))
COUNT=0

echo "========================================"
echo "  增强方式对比实验 (共 $TOTAL 条)"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

for method in "${METHODS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    COUNT=$(( COUNT + 1 ))
    echo ""
    echo "----------------------------------------"
    echo "  [$COUNT/$TOTAL] method=$method  augment_type=$aug"
    echo "  时间: $(date '+%H:%M:%S')"
    echo "----------------------------------------"

    $PYTHON main.py \
      --method "$method" \
      --augment_type "$aug" \
      --augment_shot 5 \
      --target_test \
      --cgan_version pc \
      --num_runs 3

    echo "  [$COUNT/$TOTAL] 完成: method=$method  augment_type=$aug"
  done
done

echo ""
echo "========================================"
echo "  全部完成！结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
