# MambaATM: A Framework for Automated Trading with Mamba

## Install Guide

```python

conda create -n mamba2 python=3.10.14 -y

conda activate mamba2

pip install ./mamba/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


pip install ./mamba/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


cd /home/cjh2004/Mamba3.3/pykt-toolkit


pip install -e. -i https://pypi.tuna.tsinghua.edu.cn/simple


cd ..


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# test

```py
python main.py
```
