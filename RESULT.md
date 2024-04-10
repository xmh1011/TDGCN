# 实验结果汇总

## DEAP数据集实验结果

第一轮实验：

2080ti_2上已有能正常运行的代码：

EEGNet-deap-A  EEGTCNet-deap-A  EEGTCN-WQJ-A  LGG-fro-deap-A  LGG-gen-deap-A
EEGNet-deap-D  EEGTCNet-deap-D  EEGTCN-WQJ-V  LGG-fro-deap-D  LGG-gen-deap-D
EEGNet-deap-L  EEGTCNet-deap-L  EEG-WQJ-A     LGG-fro-deap-L  LGG-gen-deap-L
EEGNet-deap-V  EEGTCNet-deap-V  EEG-WQJ-V     LGG-fro-deap-V  LGG-gen-deap-V

2080ti_1上已有正常运行代码：

LGG-hem-deap-D  TDGCN-fro-deap-A  TDGCN-gen-deap-L  TDGCN-hem-deap-D
LGG-hem-deap-L  TDGCN-fro-deap-D  TDGCN-gen-deap-A  TDGCN-hem-deap-L
LGG-hem-deap-V  TDGCN-fro-deap-L  TDGCN-gen-deap-D  TDGCN-hem-deap-V
LGG-hem-deap-A  TDGCN-fro-deap-V  TDGCN-gen-deap-V  TDGCN-hem-deap-A

| 模型/分区          | 标签 | 平均Acc | Acc标准差 | 平均F1 | F1标准差 |
|----------------|----|-------|--------|------|-------|
| EEGNet         | A  |       |        |      |       |
| EEGNet         | V  |       |        |      |       |
| EEGNet         | D  |       |        |      |       |
| EEGNet         | L  |       |        |      |       |
| DeepConvNet    | A  |       |        |      |       |
| DeepConvNet    | V  |       |        |      |       |
| DeepConvNet    | D  |       |        |      |       |
| DeepConvNet    | L  |       |        |      |       |
| ShallowConvNet | A  |       |        |      |       |
| ShallowConvNet | V  |       |        |      |       |
| ShallowConvNet | D  |       |        |      |       |
| ShallowConvNet | L  |       |        |      |       |
| EEGTCNet       | A  |       |        |      |       |
| EEGTCNet       | V  |       |        |      |       |
| EEGTCNet       | D  |       |        |      |       |
| EEGTCNet       | L  |       |        |      |       |
| RGNN           | A  |       |        |      |       |
| RGNN           | V  |       |        |      |       |
| RGNN           | D  |       |        |      |       |
| RGNN           | L  |       |        |      |       |
| R2G-STNN       | A  |       |        |      |       |
| R2G-STNN       | V  |       |        |      |       |
| R2G-STNN       | D  |       |        |      |       |
| R2G-STNN       | L  |       |        |      |       |
| HRNN           | A  |       |        |      |       |
| HRNN           | V  |       |        |      |       |
| HRNN           | D  |       |        |      |       |
| HRNN           | L  |       |        |      |       |
| LGG - fro      | A  |       |        |      |       |
| LGG - fro      | V  |       |        |      |       |
| LGG - fro      | D  |       |        |      |       |
| LGG - fro      | L  |       |        |      |       |
| LGG - gen      | A  |       |        |      |       |
| LGG - gen      | V  |       |        |      |       |
| LGG - gen      | D  |       |        |      |       |
| LGG - gen      | L  |       |        |      |       |
| LGG - hem      | A  |       |        |      |       |
| LGG - hem      | V  |       |        |      |       |
| LGG - hem      | D  |       |        |      |       |
| LGG - hem      | L  |       |        |      |       |
| AT-DGCN - fro  | A  |       |        |      |       |
| AT-DGCN  - fro | V  |       |        |      |       |
| AT-DGCN  - fro | D  |       |        |      |       |
| AT-DGCN  - fro | L  |       |        |      |       |
| AT-DGCN  - gen | A  |       |        |      |       |
| AT-DGCN  - gen | V  |       |        |      |       |
| AT-DGCN  - gen | D  |       |        |      |       |
| AT-DGCN  - gen | L  |       |        |      |       |
| AT-DGCN  - hem | A  |       |        |      |       |
| AT-DGCN  - hem | V  |       |        |      |       |
| AT-DGCN  - hem | D  |       |        |      |       |
| AT-DGCN  - hem | L  |       |        |      |       |

## 私有数据集实验结果

第一轮实验：

| 模型/分区          | 标签 | 平均Acc | Acc标准差 | 平均F1 | F1标准差 |
|----------------|----|-------|--------|------|-------|
| EEGNet         | A  |       |        |      |       |
| EEGNet         | V  |       |        |      |       |
| DeepConvNet    | A  |       |        |      |       |
| DeepConvNet    | V  |       |        |      |       |
| ShallowConvNet | A  |       |        |      |       |
| ShallowConvNet | V  |       |        |      |       |
| EEGTCNet       | A  |       |        |      |       |
| EEGTCNet       | V  |       |        |      |       |
| RGNN           | A  |       |        |      |       |
| RGNN           | V  |       |        |      |       |
| R2G-STNN       | A  |       |        |      |       |
| R2G-STNN       | V  |       |        |      |       |
| HRNN           | A  |       |        |      |       |
| HRNN           | V  |       |        |      |       |
| LGG - fro      | A  |       |        |      |       |
| LGG - fro      | V  |       |        |      |       |
| LGG - gen      | A  |       |        |      |       |
| LGG - gen      | V  |       |        |      |       |
| LGG - hem      | A  |       |        |      |       |
| LGG - hem      | V  |       |        |      |       |
| AT-DGCN - fro  | A  |       |        |      |       |
| AT-DGCN  - fro | V  |       |        |      |       |
| AT-DGCN  - gen | A  |       |        |      |       |
| AT-DGCN  - gen | V  |       |        |      |       |
| AT-DGCN  - hem | A  |       |        |      |       |
| AT-DGCN  - hem | V  |       |        |      |       |

第二轮实验：

| 模型/分区          | 标签 | 平均Acc | Acc标准差 | 平均F1 | F1标准差 |
|----------------|----|-------|--------|------|-------|
| EEGNet         | A  |       |        |      |       |
| EEGNet         | V  |       |        |      |       |
| DeepConvNet    | A  |       |        |      |       |
| DeepConvNet    | V  |       |        |      |       |
| ShallowConvNet | A  |       |        |      |       |
| ShallowConvNet | V  |       |        |      |       |
| EEGTCNet       | A  |       |        |      |       |
| EEGTCNet       | V  |       |        |      |       |
| RGNN           | A  |       |        |      |       |
| RGNN           | V  |       |        |      |       |
| R2G-STNN       | A  |       |        |      |       |
| R2G-STNN       | V  |       |        |      |       |
| HRNN           | A  |       |        |      |       |
| HRNN           | V  |       |        |      |       |
| LGG - fro      | A  |       |        |      |       |
| LGG - fro      | V  |       |        |      |       |
| LGG - gen      | A  |       |        |      |       |
| LGG - gen      | V  |       |        |      |       |
| LGG - hem      | A  |       |        |      |       |
| LGG - hem      | V  |       |        |      |       |
| AT-DGCN - fro  | A  |       |        |      |       |
| AT-DGCN  - fro | V  |       |        |      |       |
| AT-DGCN  - gen | A  |       |        |      |       |
| AT-DGCN  - gen | V  |       |        |      |       |
| AT-DGCN  - hem | A  |       |        |      |       |
| AT-DGCN  - hem | V  |       |        |      |       |


第三轮实验：

| 模型/分区          | 标签 | 平均Acc | Acc标准差 | 平均F1 | F1标准差 |
|----------------|----|-------|--------|------|-------|
| EEGNet         | A  |       |        |      |       |
| EEGNet         | V  |       |        |      |       |
| DeepConvNet    | A  |       |        |      |       |
| DeepConvNet    | V  |       |        |      |       |
| ShallowConvNet | A  |       |        |      |       |
| ShallowConvNet | V  |       |        |      |       |
| EEGTCNet       | A  |       |        |      |       |
| EEGTCNet       | V  |       |        |      |       |
| RGNN           | A  |       |        |      |       |
| RGNN           | V  |       |        |      |       |
| R2G-STNN       | A  |       |        |      |       |
| R2G-STNN       | V  |       |        |      |       |
| HRNN           | A  |       |        |      |       |
| HRNN           | V  |       |        |      |       |
| LGG - fro      | A  |       |        |      |       |
| LGG - fro      | V  |       |        |      |       |
| LGG - gen      | A  |       |        |      |       |
| LGG - gen      | V  |       |        |      |       |
| LGG - hem      | A  |       |        |      |       |
| LGG - hem      | V  |       |        |      |       |
| AT-DGCN - fro  | A  |       |        |      |       |
| AT-DGCN  - fro | V  |       |        |      |       |
| AT-DGCN  - gen | A  |       |        |      |       |
| AT-DGCN  - gen | V  |       |        |      |       |
| AT-DGCN  - hem | A  |       |        |      |       |
| AT-DGCN  - hem | V  |       |        |      |       |

第四轮实验：

| 模型/分区          | 标签 | 平均Acc | Acc标准差 | 平均F1 | F1标准差 |
|----------------|----|-------|--------|------|-------|
| EEGNet         | A  |       |        |      |       |
| EEGNet         | V  |       |        |      |       |
| DeepConvNet    | A  |       |        |      |       |
| DeepConvNet    | V  |       |        |      |       |
| ShallowConvNet | A  |       |        |      |       |
| ShallowConvNet | V  |       |        |      |       |
| EEGTCNet       | A  |       |        |      |       |
| EEGTCNet       | V  |       |        |      |       |
| RGNN           | A  |       |        |      |       |
| RGNN           | V  |       |        |      |       |
| R2G-STNN       | A  |       |        |      |       |
| R2G-STNN       | V  |       |        |      |       |
| HRNN           | A  |       |        |      |       |
| HRNN           | V  |       |        |      |       |
| LGG - fro      | A  |       |        |      |       |
| LGG - fro      | V  |       |        |      |       |
| LGG - gen      | A  |       |        |      |       |
| LGG - gen      | V  |       |        |      |       |
| LGG - hem      | A  |       |        |      |       |
| LGG - hem      | V  |       |        |      |       |
| AT-DGCN - fro  | A  |       |        |      |       |
| AT-DGCN  - fro | V  |       |        |      |       |
| AT-DGCN  - gen | A  |       |        |      |       |
| AT-DGCN  - gen | V  |       |        |      |       |
| AT-DGCN  - hem | A  |       |        |      |       |
| AT-DGCN  - hem | V  |       |        |      |       |


第五轮实验：

| 模型/分区          | 标签 | 平均Acc | Acc标准差 | 平均F1 | F1标准差 |
|----------------|----|-------|--------|------|-------|
| EEGNet         | A  |       |        |      |       |
| EEGNet         | V  |       |        |      |       |
| DeepConvNet    | A  |       |        |      |       |
| DeepConvNet    | V  |       |        |      |       |
| ShallowConvNet | A  |       |        |      |       |
| ShallowConvNet | V  |       |        |      |       |
| EEGTCNet       | A  |       |        |      |       |
| EEGTCNet       | V  |       |        |      |       |
| RGNN           | A  |       |        |      |       |
| RGNN           | V  |       |        |      |       |
| R2G-STNN       | A  |       |        |      |       |
| R2G-STNN       | V  |       |        |      |       |
| HRNN           | A  |       |        |      |       |
| HRNN           | V  |       |        |      |       |
| LGG - fro      | A  |       |        |      |       |
| LGG - fro      | V  |       |        |      |       |
| LGG - gen      | A  |       |        |      |       |
| LGG - gen      | V  |       |        |      |       |
| LGG - hem      | A  |       |        |      |       |
| LGG - hem      | V  |       |        |      |       |
| AT-DGCN - fro  | A  |       |        |      |       |
| AT-DGCN  - fro | V  |       |        |      |       |
| AT-DGCN  - gen | A  |       |        |      |       |
| AT-DGCN  - gen | V  |       |        |      |       |
| AT-DGCN  - hem | A  |       |        |      |       |
| AT-DGCN  - hem | V  |       |        |      |       |