# NKU 机器学习课程实验

实验一：自写 kNN 手写数字识别、1593 轮留一法交叉验证，以及真实 Weka IBk 对比。包含 k 值扫描、固定测试集评估、混淆矩阵、误分类图片和逐样本预测。实验报告尚未编写。

- [实验一源码](lab1/exp_1.py)
- [运行说明](lab1/README.md)
- [精度结果](lab1/results/metrics.csv)
- [Weka 对比表](lab1/results/weka_comparison.csv)
- [图表与运行日志](lab1/results/)

在仓库根目录运行：

```powershell
python -m pip install -r lab1/requirements.txt
python lab1/exp_1.py
```

重新运行实际 Weka 对比（需要 Java）：

```powershell
python lab1/prepare_weka.py
python lab1/exp_1.py --weka
```

Weka 运行库由准备脚本下载；系统缓存、Python 环境和下载的 JAR 不纳入版本控制。

数据来自 [UCI Semeion Handwritten Digit](https://archive.ics.uci.edu/dataset/178/semeion%2Bhandwritten%2Bdigit)，许可为 CC BY 4.0，引用 DOI：[10.24432/C5SC8V](https://doi.org/10.24432/C5SC8V)。课程提供的训练集与测试集合并后，全部 1593 条样本的像素和标签与官方数据一致，行顺序不同。
