# ML-Sharp



## [![Repography logo](https://images.repography.com/logo.svg)](https://repography.com)  Top contributors
[![Top contributors](https://images.repography.com/26015867/xin-pu/ML-Sharp/top-contributors/cbd147ff7d7714b2a2514c831d1c4e44_table.svg)](https://github.com/xin-pu/ML-Sharp/graphs/contributors)


## [![Repography logo](https://images.repography.com/logo.svg)](https://repography.com)  Structure
[![Structure](https://images.repography.com/26015867/xin-pu/ML-Sharp/structure/83149b229e6786964d15d69f2620e650_table.svg)](https://github.com/xin-pu/ML-Sharp)



## Demo
```
var trainDataset = GetIris("iris-train.txt");
var valDataset = GetIris("iris-test.txt");

var trainer = new GDTrainer<IrisDataOneHot>
{
    TrainDataset = trainDataset.Shuffle(),
    ValDataset = valDataset.Shuffle(),
    ModelGd = new Perceptron<IrisDataOneHot>(3),
    Optimizer = new Nadam(1E-2),
    Loss = new CategoricalCrossentropy(),

    TrainPlan = new TrainPlan {Epoch = 20, BatchSize = 10},
    Metrics = new ObservableCollection<Metric>
    {
        new CategoricalAccuracy(),
        new Metrics.Categorical.CategoricalCrossentropy()
    },

    Print = _testOutputHelper.WriteLine
};

await trainer.Fit();
print(trainer.ModelGd);

var Iris1 = new IrisDataOneHot
{
    Label = 1,
    SepalLength = 6.6,
    SepalWidth = 2.9,
    PetalLength = 4.6,
    PetalWidth = 1.3
};
var Iris2 = new IrisDataOneHot
{
    Label = 2,
    SepalLength = 7.2,
    SepalWidth = 3.5,
    PetalLength = 6.1,
    PetalWidth = 2.4
};

var pred = trainer.ModelGd.Call(Iris1);
print(pred);


pred = trainer.ModelGd.Call(Iris2);
print(pred);
 ```

 ![Demo](document/demo_log.png)


 ## MindMaster

 ![Demo](document/机器学习.png)
 
 ![Demo](document/数学知识.png)

 ![Demo](document/优化算法.png)

 ![Demo](document/一阶优化算法.png)

 ![Demo](document/监督学习.png)

 ![Demo](document/无监督学习.png)

 ![Demo](document/特征选择策略.png)

 ![Demo](document/其他.png)

