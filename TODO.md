# TODO List

- 4.10 TODO:
  - 数据更新上传至所有服务器，可以只更新sub10。
  - 代码结果汇总。

- 后续TODO:
  - 对比模型代码的修改。目前已完成EEGNet、DeepConvNet、ShallowConvNet、EEGTCNet、TSception。但是DeepConvNet、ShallowConvNet目前跑报错，推测是参数设置问题，需要设置一个合理的参数。
  - 等待实现的模型：
    - [RGNN](https://ieeexplore.ieee.org/document/9091308) [code](https://github.com/machwine/RGNN_EEG_emotiondetector)
    - [R2G-STNN](https://www.computer.org/csdl/journal/ta/2022/02/08736804/1aR7Ba3OXNm) [code](https://github.com/Ruiver/CTCNet/blob/539e55ec9fed06028379d35dfd5cd4074755ffd8/src/Unimodal_Recognition/R2G-STNN.py#L4)
    - [HRNN](https://ieeexplore.ieee.org/document/9361688) [code](https://github.com/ivoryRabbit/hrnn-pytorch/blob/main/src/model.py)
    - 准备对比的模型包括：LGGNet、EEGNet、DeepConvNet、ShallowConvNet、EEGTCNet、TSception、RGNN、R2G-STNN、HRNN。前六个模型有确切的代码，后三个模型需要具体实现。如果实现有难度，只实现其中一两个就行。
    - 代码整理。实现模型的代码框架在`models`文件夹下，每个模型的代码在`models`文件夹下的对应文件夹中。
  - 论文写作：需要完成特征提取部分的详细介绍，以及模型对比部分的详细介绍。
  - 跑实验。实验结果汇总到RESULT.md中。针对deap数据集，一共有4个标签，需要对比8-10个模型，其中LGG和ATDGCN需要跑不同大脑分区方式：'fro', 'gen', 'hem'。每个模型每个标签跑5次，取平均值并计算标准差。相当于一共4*15*5=300次实验。使用16张2080ti跑除LGG和ATDGCN的模型，每张卡每次可以跑至少两组代码，每天早晚各一轮，每天跑64组代码，大约一周时间完成对比实验。LGG和ATDGCN的模型使用4080/3090跑。
  - 针对私有数据集同理，一共2个标签，需要对比8-10个模型，每个模型每个标签跑5次，取平均值并计算标准差。相当于一共2*15*5=150次实验。使用16张2080ti跑，每天早晚各一轮，每天跑64组代码，大约四天时间完成对比实验。
  - 上述对比实验，抛开特殊情况，大约需要三周时间完成。优先跑私有数据集的实验，时间紧张deap的实验可以只跑一次。
  