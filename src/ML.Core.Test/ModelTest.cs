﻿using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using FluentAssertions;
using ML.Core.Data;
using ML.Core.Data.DataStructs;
using ML.Core.Data.Loader;
using ML.Core.Metrics;
using ML.Core.Metrics.Categorical;
using ML.Core.Metrics.Regression;
using ML.Core.Models;
using ML.Core.Optimizers;
using ML.Core.Trainers;
using Xunit;
using Xunit.Abstractions;
using CategoricalCrossentropy = ML.Core.Losses.CategoricalCrossentropy;

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
            var model = new MultipleLinearRegression();
            print(model);
            model.Weights.Should().BeNull();
        }


        [Fact]
        public void TestModel()
        {
            var path = Path.Combine(dataFolder, "iris-train.txt");
            var data = TextLoader.LoadDataSet<IrisData>(path, new[] {'\t'});

            var model = new MultipleLinearRegression();
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
        public async Task TestPerceptron()
        {
            var trainDataset = GetIris<IrisDataOneHot>("iris-train.txt");
            var valDataset = GetIris<IrisDataOneHot>("iris-test.txt");

            var trainer = new GDTrainer
            {
                TrainDataset = trainDataset.Shuffle(),
                ValDataset = valDataset.Shuffle(),
                ModelGd = new Perceptron(3),
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
        }


        [Fact]
        public async Task Save()
        {
            var trainDataset = GetIris<IrisDataOneHot>("iris-train.txt");
            var valDataset = GetIris<IrisDataOneHot>("iris-test.txt");

            var trainer = new GDTrainer
            {
                TrainDataset = trainDataset.Shuffle(),
                ValDataset = valDataset.Shuffle(),
                ModelGd = new Perceptron(3),
                Optimizer = new Nadam(1E-2),
                Loss = new CategoricalCrossentropy(),

                TrainPlan = new TrainPlan {Epoch = 100, BatchSize = 10},
                Metrics = new ObservableCollection<Metric>
                {
                    new CategoricalAccuracy(),
                    new Metrics.Categorical.CategoricalCrossentropy()
                },

                Print = _testOutputHelper.WriteLine
            };

            await trainer.Fit();
            print(trainer.ModelGd);

            trainer.ModelGd.Save("model.txt");
        }

        [Fact]
        public void Load()
        {
            var model = ModelGD.Load("model.txt");


            var Iris1 = new IrisDataOneHot
            {
                Label = 1,
                SepalLength = 6.6,
                SepalWidth = 2.9,
                PetalLength = 4.6,
                PetalWidth = 1.3
            };

            var pred = model.Call(Iris1);
            print(pred);

            var Iris2 = new IrisDataOneHot
            {
                Label = 2,
                SepalLength = 7.2,
                SepalWidth = 3.5,
                PetalLength = 6.1,
                PetalWidth = 2.4
            };

            pred = model.Call(Iris2);
            print(pred);
        }


        private Dataset<DataView> GetIris<T>(string path) where T : DataView
        {
            var trainpath = Path.Combine(dataFolder, path);
            var dataset = TextLoader.LoadDataSet<T>(trainpath, new[] {'\t'});
            return dataset;
        }


        [Fact]
        public void TestKNN()
        {
            var trainDataset = GetIris<IrisData>("iris-train.txt").ToDatasetNDarray();
            var knn = new KNN(2);
            knn.LoadDataView(trainDataset.Feature, trainDataset.Label);


            var testDataset = GetIris<IrisData>("iris-test.txt").ToDatasetNDarray();
            var Y_pred = knn.Call(testDataset.Feature);
            print(Y_pred);
            print(testDataset.Label);
            var categoricalAcc = new MeanSquaredError().Call(testDataset.Label, Y_pred);
            print(categoricalAcc);
        }
    }
}