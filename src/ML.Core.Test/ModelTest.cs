﻿using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using FluentAssertions;
using ML.Core.Data;
using ML.Core.Data.Loader;
using ML.Core.Losses.CategoricalLosses;
using ML.Core.Metrics;
using ML.Core.Metrics.Regression;
using ML.Core.Models;
using ML.Core.Optimizers;
using ML.Core.Test.DataStructs;
using ML.Core.Trainers;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test
{
    public class ModelTest : AbstractTest
    {
        private readonly string dataFolder = @"..\..\..\..\..\data";

        public ModelTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestCreateModel()
        {
            var model = new MultipleLinearRegression<IrisData>();
            print(model);
            model.Weights.Should().BeNull();
        }


        [Fact]
        public void TestModel()
        {
            var path = Path.Combine(dataFolder, "iris-train.txt");
            var data = TextLoader<IrisData>.LoadDataSet(path, splitChar: '\t');

            var model = new MultipleLinearRegression<IrisData>();
            model.PipelineDataSet(data);
            print(model);
            model.Weights.Should().NotBeNull();

            var iEnumerator = data.GetEnumerator();
            if (iEnumerator.MoveNext() && iEnumerator.Current is Dataset<IrisData> batch)
            {
                var feature = batch.ToDatasetNDarray().Feature;
                model.CallGraph(feature);
                var predict = model.Call(feature);
                print(predict);
            }
        }


        [Fact]
        public async Task TestBinaryLogicClassify()
        {
            var trainDataset = GetIris("iris-train.txt");
            var valDataset = GetIris("iris-test.txt");

            var trainer = new GDTrainer<IrisDataOneHot>
            {
                TrainDataset = trainDataset.Shuffle(),
                ValDataset = valDataset.Shuffle(),
                ModelGd = new Perceptron<IrisDataOneHot>(3),
                Optimizer = new Adam(1E-2),
                Loss = new CategoricalCrossentropy(),

                TrainPlan = new TrainPlan {Epoch = 100, BatchSize = 25},
                Metrics = new ObservableCollection<Metric> {new MeanAbsoluteError()},

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
        }


        private Dataset<IrisDataOneHot> GetIris(string path)
        {
            var trainpath = Path.Combine(dataFolder, path);
            var dataset = TextLoader<IrisDataOneHot>.LoadDataSet(trainpath, true, '\t');
            return dataset;
        }
    }
}