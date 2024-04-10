# ATDGCN

跑TSception模型的时候，参数为：
```python
    parser.add_argument('--input-shape', type=tuple, default=(1, 28, 512))
    parser.add_argument('--model', type=str, default='TSception', choices=['LGGNet', 'ATDCGN', 'EEGNet', 'DeepConvNet', 'ShallowConvNet', 'EEGTCNet', 'MBEEG_SENet', 'EEGNetClassifier', 'TSception'])
    parser.add_argument('--T', type=int, default=15)
    parser.add_argument('--graph-type', type=str, default='TS', choices=['fro', 'gen', 'hem', 'BL', 'TS'])
```

```shell
conda create --name ATGCN python=3.8
```

Activate the virtual environment by:

```shell
conda activate ATGCN
```

Install the requirements by:

```shell
pip3 install -r requirements.txt
```

# Run the code
Please download the DEAP dataset at [this website](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/). Please place the "data_preprocessed_python" folder in the same location as the script. To run the code for the emotion (valence) classification, please type the following command in the terminal:

```shell
python3 main.py --data-path '/Users/xiaominghao/code/EEG-Music/eeg/' --label-type 'V' --graph-type 'gen'
```

To run the code for preference (liking) classification, please type the following command in the terminal:

```shell
python3 main.py --data-path './Users/xiaominghao/code/EEG-Music/deap_format/' --label-type 'L' --graph-type 'hem'
```

The results will be saved into "result_DEAP.txt" located at './save/result/'. 

# Reproduce the results
We highly suggest running the code on a Ubuntu 18.04 or above machine using anaconda with the provided requirements to reproduce the results. 
You can also download the saved model at [this website](https://drive.google.com/file/d/12lIbX6ti7cDCv3mVDY7TTd4QIc2cNEYE/view?usp=sharing) to reproduce the results in the paper. After extracting the downloaded "save.zip", please place it at the same location as the scripts, and run the code by:

```shell
python3 main.py --data-path './data_preprocessed_python/' --label-type 'V' --graph-type 'gen' --reproduce
```
