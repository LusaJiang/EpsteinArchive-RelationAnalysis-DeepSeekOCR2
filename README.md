# 山东农业大学通用课程大作业模板

这是一个专为山东农业大学学生设计的LaTeX课程报告模板，适用于各种课程的平时论文和期末论文写作。

## 📋 项目简介

本模板基于LaTeX开发，提供标准化的学术论文格式，包含完整的封面、目录、章节结构等功能。支持中文排版，预设了合适的字体、字号和页边距。

### 主要特性
- ✅ 标准化论文格式（页边距2.5cm）
- ✅ 中文字体支持（宋体）和英文字体支持（Times New Roman）
- ✅ 自动生成封面和目录
- ✅ 完整的数学公式支持
- ✅ 图片、表格插入功能
- ✅ 参考文献管理（BibTeX）
- ✅ 自定义文本框样式
- ✅ 页眉页脚设置
- ✅ 🆕 **自动PDF编译发布**（通过GitHub Actions）

## 🚀 快速开始

### 编译要求
- XeLaTeX 编译器
- 完整的TeX发行版（如TeX Live或MiKTeX）

### 手动编译步骤
使用以下编译顺序：
```bash
xelatex main.tex
bibtex main.aux
xelatex main.tex
xelatex main.tex
```

或者使用自动化工具：
```bash
latexmk -xelatex main.tex
```

也可以使用我们提供的编译脚本：
```bash
chmod +x compile.sh
./compile.sh
```

## 🤖 自动PDF编译与发布

详见 [release-pdf.yml](./.github/workflows/release-pdf.yml).

### 使用方法

#### 方法一：命令行操作
```bash
# 1. 提交所有更改
git add .
git commit -m "准备发布新版本"

# 2. 创建并推送标签（标签名必须以v开头）
git tag v1.0.0
git push origin v1.0.0
```

#### 方法二：GitHub网页操作
1. 在GitHub仓库页面点击"Releases"
2. 点击"Draft a new release"
3. 选择或创建一个以`v`开头的标签（如`v1.0.0`）
4. 填写发布说明
5. 点击"Publish release"

### 发布结果
等待约2-5分钟后，即可在GitHub的[Releases](../../releases)页面看到：
- 自动生成的PDF文件（如：`山东农业大学课程报告_1.0.0.pdf`）
- 详细的发布说明
- 编译环境和时间信息

## 📁 项目结构

```
├── main.tex          # 主文档文件
├── SDAUReport.sty    # 模板样式文件
├── reference.bib     # 参考文献数据库
├── .gitignore        # Git忽略文件配置
├── .github/
│   └── workflows/
│       ├── release-pdf.yml         # PDF自动编译发布
├── figures/          # 图片资源文件夹
│   ├── sdau_logo.png        # 山东农业大学校徽
│   └── sdau_logo_notitle.png # 无标题校徽
└── README.md         # 项目说明文档
```

## 🛠️ 使用指南

### 1. 初始设置
首次使用时，请编辑 `SDAUReport.sty` 文件中的个人信息：

```latex
% 修改以下信息
\author{
\vspace{0.5cm}
\kaishu\Large 学院\ \dlmu[9cm]{你的学院名称} \\
\vspace{0.5cm}
\kaishu\Large 班级\ \dlmu[9cm]{你的班级} \\
\vspace{0.5cm}
\kaishu\Large 学号\ \dlmu[9cm]{你的学号} \\
\vspace{0.5cm}
\kaishu\Large 姓名\ \dlmu[9cm]{你的姓名} \\
}
```

同时修改标题信息：
```latex
\title{ 
\vspace{1cm}
\heiti \Huge \textbf{{你的课程名称}} \par
\vspace{1cm} 
\heiti \Large {\underline{你的报告题目}}
\vspace{2cm}
}
```

### 2. 页眉设置
在 `SDAUReport.sty` 中修改页眉内容：
```latex
\fancyhead[C]{\fangsong 你的页眉内容}
```

### 3. 添加内容
在 `main.tex` 中添加你的报告内容：

#### 插入章节
```latex
\section{章节标题}
这里是章节内容...
```

#### 插入公式
```latex
% 行内公式
欧拉公式 $v-\varepsilon+\phi=2$

% 行间公式
\begin{equation}
    v-\varepsilon+\phi=2
    \label{Euler}
\end{equation}
```

#### 插入图片
```latex
\begin{figure}[!htbp]
    \centering
    \includegraphics[width=.5\textwidth]{figures/图片文件名.png}
    \caption{图片说明}
    \label{fig:label}
\end{figure}
```

#### 插入表格
```latex
\begin{table}[!htbp]
    \centering
    \begin{tabular}{l|l}
        \hline
        列1 & 列2 \\
        \hline
        内容1 & 内容2 \\
        \hline
    \end{tabular}
    \caption{表格说明}
    \label{tab:label}
\end{table}
```

#### 插入参考文献
在 `reference.bib` 中添加文献条目，然后在正文中引用：
```latex
文献\cite{文献标签}表明...
```

### 4. 自定义功能

#### 文本框
使用 `\tbox{}` 命令插入圆角灰底文本框：
```latex
\tbox{这是文本框内容}
```

#### 定理环境
模板提供了多种数学环境：
```latex
\begin{Theorem}
定理内容
\end{Theorem}

\begin{Definition}
定义内容
\end{Definition}

\begin{Example}
例子内容
\end{Example}
```

## 📚 参考文献格式

参考文献使用BibTeX管理，在 `reference.bib` 文件中按照标准格式添加：

```bibtex
@article{example2023,
  title={文章标题},
  author={作者姓名},
  journal={期刊名称},
  year={2023}
}

@inproceedings{conf2023,
  title={会议论文标题},
  author={作者姓名},
  booktitle={会议名称},
  pages={1--10},
  year={2023}
}
```

## 🔧 常见问题

### Q: 编译时报错找不到字体？
A: 确保安装了完整的TeX发行版，并且系统中有相应的中文字体。

### Q: 图片无法显示？
A: 检查图片路径是否正确，建议将图片放在 `figures` 文件夹中。

### Q: 参考文献不显示？
A: 确保按正确的编译顺序执行，并检查 `.bib` 文件格式是否正确。

### Q: 如何修改页面边距？
A: 在 `SDAUReport.sty` 中修改 geometry 包的参数：
```latex
\RequirePackage[left=2.50cm, right=2.50cm, top=2.50cm, bottom=2.50cm]{geometry}
```

### Q: 自动编译失败怎么办？
A: 检查以下几点：
1. 标签名是否以`v`开头
2. LaTeX源文件是否有语法错误
3. 查看Actions页面的具体错误信息
4. 可以下载编译日志进行调试

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个模板！

## 📄 许可证

本项目基于 [CC-BY-SA-4.0](./LICENSE) 许可证开源。

## 🔗 相关链接

- [GitHub仓库](https://github.com/LusaJiang/SADU_Course_Template_Latex)
- [Overleaf在线编辑](https://www.overleaf.com/latex/)
- [LaTeX入门教程](https://www.latex-project.org/help/documentation/)

---

**注意**：请根据具体课程要求调整格式规范，此模板仅供参考使用。
