# 吴恩达《深度学习专项课程》学习记录

本项目用于记录我学习吴恩达（Andrew Ng）在 Coursera 开设的 [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) 时使用的课程资料、编程作业、测验和个人笔记。

课程由 [DeepLearning.AI](https://www.deeplearning.ai/) 提供，授课教师为 [吴恩达](https://www.andrewng.org/)。项目中的大部分原始课程代码和资料来自 [amanchadha/coursera-deep-learning-specialization](https://github.com/amanchadha/coursera-deep-learning-specialization)，并在此基础上加入了个人学习笔记、运行记录和部分 PyTorch 改写内容。

## 项目内容

本项目包含深度学习专项课程的五门课程：

1. 神经网络与深度学习；
2. 改善深层神经网络：超参数调试、正则化与优化；
3. 结构化机器学习项目；
4. 卷积神经网络；
5. 序列模型。

每门课程目录中可能包含：

- Jupyter Notebook 编程作业；
- 作业所需的数据集、辅助代码和预训练模型；
- 每周测验的 Markdown 或 PDF 版本；
- 课程总结和示意图；
- 不同年份或不同框架版本的作业。

## 个人笔记

个人整理的中文笔记位于 [note.md](note.md)，目前包括：

- 逻辑回归的前向传播、反向传播和向量化；
- 多层神经网络与梯度下降公式；
- 偏差、方差和正则化；
- Mini-batch、Momentum、RMSprop 和 Adam；
- Batch Normalization、Softmax 和超参数调试；
- 机器学习策略、误差分析和数据分布不匹配；
- 卷积、池化、参数共享和稀疏连接；
- LeNet、AlexNet、VGG、ResNet 和 Inception；
- 迁移学习与数据增强。

笔记中的公式、计算示例和流程图主要用于复习课程概念。如发现错误，欢迎通过 Issue 指出。

## 课程目录

### 第一课：神经网络与深度学习

目录：[C1 - Neural Networks and Deep Learning](<C1 - Neural Networks and Deep Learning>)

主要内容：

- Python 与 NumPy 基础；
- 逻辑回归；
- 单隐藏层神经网络；
- 深层神经网络的前向传播和反向传播；
- 使用深层神经网络完成图像分类。

### 第二课：改善深层神经网络

目录：[C2 - Improving Deep Neural Networks](<C2 - Improving Deep Neural Networks Hyperparameter tuning, Regularization and Optimization>)

主要内容：

- 参数初始化；
- L2 正则化和 Dropout；
- 梯度检查；
- Mini-batch Gradient Descent；
- Momentum、RMSprop 和 Adam；
- 超参数调试；
- Batch Normalization；
- TensorFlow 基础。

### 第三课：结构化机器学习项目

目录：[C3 - Structuring Machine Learning Projects](<C3 - Structuring Machine Learning Projects>)

这门课程没有常规编程作业，重点是通过案例学习：

- 如何选择评价指标；
- 如何划分训练集、开发集和测试集；
- 如何判断 Bias、Variance 和 Data Mismatch；
- 如何进行 Error Analysis；
- 迁移学习、多任务学习和端到端学习。

### 第四课：卷积神经网络

目录：[C4 - Convolutional Neural Networks](<C4 - Convolutional Neural Networks>)

主要内容：

- 卷积和池化的 NumPy 实现；
- 使用深度学习框架构建 CNN；
- ResNet 和迁移学习；
- YOLO 目标检测；
- U-Net 图像分割；
- 人脸识别；
- 神经风格迁移。

为了配合当前以 PyTorch 为主的学习方式，本项目增加了以下 notebook：

- [卷积神经网络应用：PyTorch 版本](<C4 - Convolutional Neural Networks/Week 1/Convolution_model_Application_PyTorch.ipynb>)

该版本保持原 TensorFlow 作业的主题顺序和教学目标，使用 `torch.nn.Sequential`、自定义 `nn.Module`、`DataLoader` 和显式训练循环完成对应任务。

### 第五课：序列模型

目录：[C5 - Sequence Models](<C5 - Sequence Models>)

主要内容：

- RNN 和 LSTM；
- 字符级语言模型；
- 音乐生成；
- Word Embedding；
- Emojify 文本分类；
- Attention；
- 机器翻译；
- 触发词检测；
- Transformer、命名实体识别和问答系统。

## 编程作业与版本选择

部分作业存在多个版本。通常建议优先学习文件名中版本号较新的 notebook，例如 `v2a`、`v3a` 或带有较新年份的版本；旧版本可以用于对比 API 和课程内容变化。

以第四课第一周为例，推荐顺序为：

```text
Convolution_model_Step_by_Step_v2a.ipynb
        ↓
理解卷积和池化的底层计算
        ↓
Convolution_model_Application_PyTorch.ipynb
        ↓
使用 PyTorch 构建和训练完整 CNN
```

TensorFlow 1.x 风格的 `placeholder`、`Session` 和 `tf.contrib` 等代码可以作为历史参考，不建议作为新项目的主要写法。

## 环境配置

建议使用独立的 Python 虚拟环境：

```bash
python -m venv .venv
```

激活虚拟环境后，根据正在学习的作业安装所需依赖。常见依赖包括：

```bash
pip install numpy matplotlib pandas scipy h5py jupyter pillow
```

运行 PyTorch 版本作业前，需要根据自己的操作系统和 CUDA 环境安装 PyTorch。安装方式请参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)。

项目中的 [setup.sh](setup.sh) 用于下载部分预训练 VGG-19 数据，并解压某些作业需要的预训练模型和数据集。运行之前建议先检查脚本内容和下载地址。

## 运行 Notebook

在项目根目录启动 Jupyter：

```bash
jupyter notebook
```

也可以使用 JupyterLab、VS Code 或 PyCharm 打开 `.ipynb` 文件。

运行作业时请注意：

- 不同年份的 notebook 可能依赖不同版本的 TensorFlow、Keras 或其他库；
- PyTorch 默认使用 `NCHW`，即 `batch × channels × height × width`；
- 部分原始 TensorFlow 作业使用 `NHWC`；
- 训练大型模型前请确认内存和显存是否足够；
- 某些 notebook 已保存输出，文件体积可能较大。

## 大文件与 Git LFS

项目包含 HDF5 数据集、模型文件和其他二进制资源。部分文件通过 Git LFS 管理，相关规则位于 [.gitattributes](.gitattributes)。

克隆项目后，如发现 LFS 文件只有几行指针文本，请安装 Git LFS 并执行：

```bash
git lfs install
git lfs pull
```

当前仓库中的单个普通文件均未超过 GitHub 的 100 MiB 限制，因此课程 PDF、图片和 notebook 被完整保留。IDE 配置、虚拟环境、Python 缓存和 Jupyter 临时文件由 [.gitignore](.gitignore) 排除。

## 资料来源与致谢

感谢以下课程和项目提供的教学内容：

- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)；
- [DeepLearning.AI](https://www.deeplearning.ai/)；
- [Andrew Ng](https://www.andrewng.org/)；
- [amanchadha/coursera-deep-learning-specialization](https://github.com/amanchadha/coursera-deep-learning-specialization)；
- [Aman Chadha 的课程笔记](https://aman.ai/)。

除特别说明外，课程题目、图示和原始作业代码的版权归原作者及课程提供方所有。本仓库中的个人笔记和 PyTorch 改写内容仅用于学习交流。

## 免责声明

本仓库中的作业答案和测验资料仅供学习、复习与排错参考。建议先独立完成课程作业和测验，在遇到困难时再查看相关实现。

请不要直接复制代码提交到课程平台，也不要将本仓库用于违反 Coursera、DeepLearning.AI 或原作者学术诚信要求的用途。真正重要的是理解代码背后的计算过程、模型设计和调试方法。
