﻿using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using ML.Core.Data;
using ML.Core.Data.DataStructs;
using ML.Core.Data.Loader;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Metrics.Regression;
using ML.Core.Models;
using ML.Core.Optimizers;
using ML.Core.Trainers;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test
{
    public class TrainerTest : AbstractTest
    {
        private readonly string dataFolder = @"..\..\..\..\..\data";

        public TrainerTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestDataSet()
        {
            var path = Path.Combine(dataFolder, "data_singlevar.txt");
            var Dataset = TextLoader.LoadDataSet<LinearData>(path, ',', false);
            print(Dataset);
        }

        [Fact]
        public async Task TestMultipleLinearRegression()
        {
            var path = Path.Combine(dataFolder, "data_singlevar.txt");

            var trainer = new GDTrainer
            {
                TrainDataset = TextLoader.LoadDataSet<LinearData>(path, ',', false).Shuffle(),
                ModelGd = new MultipleLinearRegression(),
                Optimizer = new Nesterov(1E-2),
                Loss = new MeanSquared(),
                TrainPlan = new TrainPlan {Epoch = 100, BatchSize = 25},
                Metrics = new ObservableCollection<Metric>
                {
                    new RSquared()
                },
                Print = _testOutputHelper.WriteLine
            };

            await trainer.Fit();
            print(trainer.ModelGd);
        }


        [Fact]
        public async Task TestPolyRegression()
        {
            var path = Path.Combine(dataFolder, "data_singlevar.txt");

            var trainer = new GDTrainer
            {
                TrainDataset = TextLoader.LoadDataSet<LinearData>(path, ',', false),
                ModelGd = new PolynomialRegression(),
                Optimizer = new Momentum(1E-1),
                Loss = new MeanSquared(),
                TrainPlan = new TrainPlan {Epoch = 100, BatchSize = 50},
                Metrics = new ObservableCollection<Metric> {new MeanAbsoluteError()},
                Print = _testOutputHelper.WriteLine
            };

            await trainer.Fit();
            print(trainer.ModelGd);
        }


        [Fact]
        public async Task TestBinaryLogicClassify()
        {
            var trainDataset = GetBinaryIris("iris-train.txt");
            var valDataset = GetBinaryIris("iris-test.txt");

            var trainer = new GDTrainer
            {
                TrainDataset = trainDataset.Shuffle(),
                ValDataset = valDataset.Shuffle(),
                ModelGd = new BinaryLogicClassify(),
                Optimizer = new Momentum(1E-1),
                Loss = new BinaryCrossentropy(),

                TrainPlan = new TrainPlan {Epoch = 100, BatchSize = 25},
                Metrics = new ObservableCollection<Metric> {new MeanAbsoluteError()},

                Print = _testOutputHelper.WriteLine
            };

            await trainer.Fit();
            print(trainer.ModelGd);
            var pred = trainer.ModelGd.Call(valDataset.ToDatasetNDarray().Feature);
            print(pred);
        }


        private Dataset<DataView> GetBinaryIris(string path)
        {
            var trainpath = Path.Combine(dataFolder, path);
            var dataset = TextLoader.LoadDataSet<IrisData>(trainpath, '\t');
            var some = dataset.Value;
            dataset = new Dataset<DataView>(some);
            return dataset;
        }
    }
}