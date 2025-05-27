# 项目名称

基于Nvidia ncu的roofline分析示例。

## 目录

- [安装](#安装)
- [使用方法](#使用方法)
- [示例](#示例)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [致谢](#致谢)

## 安装

```bash
# 克隆仓库
git clone https://github.com/your_username/your_project.git
cd your_project

# 安装主要依赖
pip install -r requirements.txt

# 安装ncu

```

在主机上安装ncu用于可视化。

## 使用方法

见python脚本的帮助信息。

## 示例

```bash
ncu --set full -o ncu_profile_simple python3 ncu_profile_simple.py --m 4096 --n 4096 --k 4096
```

使用主机打开生成的`ncu_profile_simple.ncu-rep`进行可视化。
![](https://raw.githubusercontent.com/Senbon-Sakura/Picture/master/202505271600204.png)

![GEMM roofline](https://raw.githubusercontent.com/Senbon-Sakura/Picture/master/202505271601196.png)
GEMM roofline curve




## 贡献指南

欢迎贡献！

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

## 致谢

感谢所有贡献者和开源社区的支持。