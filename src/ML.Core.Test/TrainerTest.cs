using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ML.Core.Data;
using ML.Core.Data.Loader;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Metrics.Regression;
using ML.Core.Models;
using ML.Core.Optimizers;
using ML.Core.Test.DataStructs;
using ML.Core.Trainers;
using Numpy;
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
                Metrics = new ObservableCollection<Metric> {new MAE()},
                Print = _testOutputHelper.WriteLine
            };

            await trainer.Fit();
            print(trainer.Model);
        }

        [Fact]
        public async Task TestPolyRegression()
        {
            var path = Path.Combine(dataFolder, "data_singlevar.txt");

            var trainer = new Trainer<LinearData>
            {
                TrainDataset = TextLoader<LinearData>.LoadDataSet(path, false),
                Model = new PolynomialRegression<LinearData>(2),
                Optimizer = new Momentum(1E-2),
                Loss = new MeanSquaredError(regularization: Regularization.L2),
                TrainPlan = new TrainPlan {Epoch = 100, BatchSize = 25},
                Metrics = new ObservableCollection<Metric> {new MAE()},
                Print = _testOutputHelper.WriteLine
            };

            await trainer.Fit();
            print(trainer.Model);
        }


        [Fact]
        public async Task TestBinaryLogicClassify()
        {
            var trainDataset = GetBinaryIris("iris-train.txt");
            var valDataset = GetBinaryIris("iris-test.txt");

            var trainer = new Trainer<IrisData>
            {
                TrainDataset = trainDataset,
                ValDataset = valDataset,
                Model = new BinaryLogicClassify<IrisData>(),
                Optimizer = new Momentum(1E-2),
                Loss = new BinaryCrossentropy(),

                TrainPlan = new TrainPlan {Epoch = 100, BatchSize = 25},
                Metrics = new ObservableCollection<Metric> {new MAE()},

                Print = _testOutputHelper.WriteLine
            };

            await trainer.Fit();
            print(trainer.Model);
            var pred = trainer.Model.Call(valDataset.ToDatasetNDarray().Feature);
            print(pred);
        }


        [Fact]
        public void Test()
        {
            var a = np.random.rand(4, 3);
            print(a);
            print(np.expand_dims(np.argmax(a, -1), -1));
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