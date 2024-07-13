# AI-Agent

### 1.创建环境、下载包
```shell
conda create agent python==3.12
```
```shell
pip install -r requirements.txt
```

### 2.安装大语言模型框架Ollama，并下载模型llama3
#### （1）安装 Ollama：https://ollama.com/
#### （2）安装Ollama后，win+R打开cmd下载模型，代码如下：
```shell
ollama pull llama3
```
Tips：可以选其他模型（如通义千问），在官网找，https://ollama.fan/getting-started/，同样是pull 
如果终端显示'Ollama不是内部或外部命令，也不是可运行的程序或批处理文件。',
则添加Ollama的路径到系统环境变量Path

### 3.运行myagent2.py
#### （1）在pycharm终端里面运行Ollama，(记得关闭代理）代码如下：
```shell
ollama serve
```
#### （2）运行myagent2.py


### 4. 运行可视化界面
如果要运行可视化的界面的话，要运行app.py文件
#### （1）在pycharm终端里面运行Ollama，(记得关闭代理）代码如下：
```shell
ollama serve
```
#### （2）打开另外一个pycharm终端，输入以下代码
```shell
streamlit run app.py
```
