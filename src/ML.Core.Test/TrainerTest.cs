using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ML.Core.Data;
using ML.Core.Data.Loader;
using ML.Core.Losses;
using ML.Core.Models;
using ML.Core.Optimizers;
using ML.Core.Test.DataStructs;
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
            var Dataset = TextLoader<LinearData>.LoadDataSet(path, false);
            print(Dataset);
        }

        [Fact]
        public async Task TestMultipleLinearRegression()
        {
            var path = Path.Combine(dataFolder, "data_singlevar.txt");

            var trainer = new Trainer<LinearData>
            {
                TrainDataset = TextLoader<LinearData>.LoadDataSet(path, false),
                Model = new MultipleLinearRegression<LinearData>(),
                Optimizer = new Momentum(1E-2),
                Loss = new MeanSquaredError(),
                TrainPlan = new TrainPlan {Epoch = 100, BatchSize = 25},
                Print = _testOutputHelper.WriteLine
            };

            await trainer.Fit();
            print(trainer.Model);
        }


        [Fact]
        public async Task TestBinaryLogicClassify()
        {
            var trainDataset = GetBinaryIris("iris-train.txt");
            var testDataset = GetBinaryIris("iris-train.txt");

            var trainer = new Trainer<IrisData>
            {
                TrainDataset = trainDataset,
                TestDataset = testDataset,
                Model = new BinaryLogicClassify<IrisData>(),
                Optimizer = new Momentum(1E-2),
                Loss = new BinaryCrossentropy(),
                TrainPlan = new TrainPlan {Epoch = 100, BatchSize = 25},
                Print = _testOutputHelper.WriteLine
            };

            await trainer.Fit();
            print(trainer.Model);
        }

        private Dataset<IrisData> GetBinaryIris(string path)
        {
            var trainpath = Path.Combine(dataFolder, path);
            var dataset = TextLoader<IrisData>.LoadDataSet(trainpath, true, '\t');
            var some = dataset.Value.Where(a => a.Label == 0 || Math.Abs(a.Label - 1) < 0.1).ToList();
            dataset = new Dataset<IrisData>(some);
            return dataset;
        }
    }
}