# TENG 鞋底摩擦纳米发电机：跑/跳/走三分类代码（纯 Python 标准库）

你可以直接运行，不依赖 `numpy/pandas/sklearn/matplotlib`。

## 1) 快速运行

```bash
python activity_classifier_teng.py --out-dir outputs
```

如果没有提供真实数据，脚本会自动生成合成 TENG 信号来演示完整流程。

输出文件（多张可视化图）：
- `outputs/confusion_matrix.svg`（混淆矩阵计数图）
- `outputs/confusion_matrix_normalized.svg`（归一化混淆矩阵）
- `outputs/class_distribution.svg`（类别样本量柱状图）
- `outputs/signal_examples.svg`（各类信号示例图）
- `outputs/feature_scatter_rms_zcr.svg`（RMS-ZCR 特征散点图）
- `outputs/feature_importance.svg`（特征重要性 Top10 条形图）
- `outputs/prediction_confidence_hist.svg`（预测置信度分布图）
- `outputs/metrics.json`（准确率、样本数、输出路径）

## 2) 用你的真实数据训练/评估

目录结构：

```text
data_root/
  walk/*.csv
  run/*.csv
  jump/*.csv
```

CSV 默认需要有 `voltage` 列（可改，例如你的列名是 `Volt`）：

```bash
python activity_classifier_teng.py \
  --data-dir data_root \
  --signal-column Volt \
  --out-dir outputs_real
```

## 3) 方案说明

- 特征：均值、标准差、RMS、峰峰值、过零率、分位数、简化频谱质心/熵等。
- 模型：纯 Python 实现的高斯朴素贝叶斯。
- 输出：自动保存多张图表用于证明区分效果，不再仅有混淆矩阵。
