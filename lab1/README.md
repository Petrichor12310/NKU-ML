# 实验一运行说明

`exp_1.py` 包含自写 kNN、留一法、k 值扫描、独立测试集评估、Weka 对比及图表导出。暂未编写实验报告。

在项目根目录 `NKU-ML` 运行：

```powershell
python -m pip install -r lab1/requirements.txt
python lab1/exp_1.py
```

程序会自动寻找实验数据目录中的 `semeion_train.txt` 和 `semeion_test.txt`，忽略 `__MACOSX`。无需修改原文件或目录名，也不依赖命令行当前目录查找数据。

实际 Weka 对比：

```powershell
python lab1/prepare_weka.py
python lab1/exp_1.py --weka
```

Weka 3.8.6 的运行库下载到 `lab1/tools/weka/`，校验 Maven 提供的摘要，来源和 SHA256 保存在 `download_manifest.json`。当前机器使用已有 MATLAB 内的 Java 8 运行时，不修改 MATLAB 或系统配置。换一台机器可安装 Java，设置 `JAVA_HOME`，或显式传入路径：

```powershell
python lab1/exp_1.py --weka --java "C:/path/to/java.exe"
```

也可使用自己的 Weka JAR 和数据路径：

```powershell
python lab1/exp_1.py --weka --weka-jar "C:/path/to/weka.jar"
python lab1/exp_1.py --train "C:/path/to/semeion_train.txt" --test "C:/path/to/semeion_test.txt" --output lab1/results_custom
python lab1/exp_1.py --tie-break random --seed 42 --output lab1/results_random
```

默认扫描 `k=1,3,5,7,9,11,13,15`，可通过 `--ks 1 3 5` 缩小范围。1、3、5 是必做值，始终保留。

输出位于 `lab1/results/`：

| 文件 | 用途 |
| --- | --- |
| `metrics.csv` | 全量 LOO、训练集内 LOO 和固定测试集的准确率 |
| `weka_comparison.csv` | 自写与真实 Weka 的对照，差异单位为百分点 |
| `weka_k1.txt`、`weka_k3.txt`、`weka_k5.txt` | Weka 的完整命令、实际输出、分类明细及混淆矩阵 |
| `loo_predictions.csv`、`holdout_predictions.csv` | 逐样本预测，行号从 0 开始 |
| `per_class_metrics.csv` | 各数字的精确率、召回率、样本数 |
| `run_metadata.json` | 数据摘要、参数、库版本、重复检查和训练集选出的 k |
| `semeion_full.arff`、`semeion_train.arff`、`semeion_test.arff` | 可直接供 Weka 打开的数据，标签是单列名义类别 |
| `digit_samples.png`、`accuracy_vs_k.png` | 数据预览和 k 值曲线 |
| `confusion_*.png`、`confusion_*.csv` | 混淆矩阵；行是真实类别，列是预测类别 |
| `loo_errors.png`、`holdout_errors.png` | 最多展示 12 个误分类样本 |
| `weka_comparison.png` | 实际 Weka 精度对照图 |

实现约定：只读取前 256 列作为特征，将后 10 列独热标签转换为数字 0–9；使用平方欧氏距离排序，与欧氏距离排序相同。默认严格选 k 个邻居，等距按训练行号排序，投票平局选较小数字。`--tie-break random` 可启用固定随机种子的投票平局处理。

全量 LOO 将 1293 条训练样本与 300 条测试样本合并，每轮仅排除被留出的行。距离矩阵只涉及像素、复用不会引入测试标签。固定测试集评估只用训练文件训练；k 由训练文件内部的 LOO 选择，测试集上的扫描结果仅供观察。默认选出 k=1，固定测试集为 273/300。

Weka 使用 `IBk -K k -x 1593` 做 1593 折交叉验证；`-X` 是 IBk 内部选择 k 的选项，不用于本次对照。Weka 会保留第 k 个距离处的全部等距邻居，自写基础算法严格选 k 个。对照 CSV 额外给出保留全部边界等距邻居的诊断结果：本次三个 k 的诊断精度均与真实 Weka 一致，支持等距处理是差异来源。文档中的约 98% 是示例，本项目保留实际结果。日志中的 MTJ 资源提示不影响本次不使用矩阵工具包的 IBk 分类。

验证命令：

```powershell
python -m unittest discover -s lab1 -p test_exp_1.py -v
```

8 项测试覆盖直接欧氏距离逐轮 LOO 的独立参考实现、自身排除与重复行、距离平局、投票平局、随机复现、无符号像素溢出、非法 k、标签与 ARFF。另已用独立 SciPy 距离函数核对全部 1593 行在 k=1、3、5 下的预测，均一致；SciPy 仅用于开发核验，不是运行依赖。

本项目算法代码为独立实现；Weka 是对比运行库。相关官方接口文档：

- [Weka IBk](https://weka.sourceforge.io/doc.stable/weka/classifiers/lazy/IBk.html)
- [Weka Evaluation](https://weka.sourceforge.io/doc.stable/weka/classifiers/Evaluation.html)
- [Weka Maven 发布说明](https://waikato.github.io/weka-wiki/maven/)
