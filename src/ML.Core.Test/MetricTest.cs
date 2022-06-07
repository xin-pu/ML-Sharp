using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using FluentAssertions;
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
    public class MetricTest : AbstractTest
    {
        public MetricTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestMeanSquaredError()
        {
            var meanSquaredError = new MeanSquaredError();
            var metrix = meanSquaredError.Call(
                np.array(new double[,] {{0, 1}, {0, 0}}),
                np.array(new double[,] {{1, 1}, {0, 0}}));
            metrix.Should().Be(0.25);
        }

        [Fact]
        public void TestMeanAbsoluteError()
        {
            var meanAbsoluteError = new MeanAbsoluteError();
            var metrix = meanAbsoluteError.Call(
                np.array(new double[,] {{0, 1}, {0, 0}}),
                np.array(new double[,] {{1, 1}, {0, 0}}));
            metrix.Should().Be(0.25);
        }

        [Fact]
        public void TestMeanSquaredLogarithmicError()
        {
            var meanAbsoluteError = new MeanSquaredLogarithmicError();
            var metrix = meanAbsoluteError.Call(
                np.array(new double[,] {{0, 1}, {0, 0}}),
                np.array(new double[,] {{1, 1}, {0, 0}}));
            metrix.Should().BeInRange(0.1201, 0.1202);
        }

        [Fact]
        public void TestMeanAbsolutePercentageError()
        {
            var meanAbsoluteError = new MeanAbsolutePercentageError();
            var metrix = meanAbsoluteError.Call(
                np.array(new double[,] {{1, 1}, {1, 1}}),
                np.array(new double[,] {{1, 1}, {0, 0}}));
            print(metrix);
        }

        [Fact]
        public void TestLogCoshError()
        {
            var meanAbsoluteError = new LogCoshError();
            var metrix = meanAbsoluteError.Call(
                np.array(new double[,] {{0, 1}, {0, 0}}),
                np.array(new double[,] {{1, 1}, {0, 0}}));
            print(metrix);
        }

        [Fact]
        public void TestRSquared()
        {
            var meanAbsoluteError = new RSquared();
            var metrix = meanAbsoluteError.Call(
                np.array(new double[,] {{0, 1}, {0, 0}}),
                np.array(new double[,] {{1, 1}, {0, 0}}));
            print(metrix);
        }

        [Fact]
        public void TestExplainedVariance()
        {
            var meanAbsoluteError = new ExplainedVariance();
            var metrix = meanAbsoluteError.Call(
                np.array(new double[,] {{0, 1}, {0, 0}}),
                np.array(new double[,] {{1, 1}, {0, 0}}));
            print(metrix);
        }

        [Fact]
        public void TestMeanRelativeError()
        {
            var meanRelativeError = new MeanRelativeError();
            var metrix = meanRelativeError.Call(
                np.array(1, 3, 2, 3),
                np.array(2, 4, 6, 8));
            metrix.Should().Be(1.25);
        }


        [Fact]
        public async Task TestMultiMetrics()
        {
            var dataFolder = @"..\..\..\..\..\data";
            var path = Path.Combine(dataFolder, "data_singlevar.txt");

            var trainer = new GDTrainer<LinearData>
            {
                TrainDataset = TextLoader<LinearData>.LoadDataSet(path, false).Shuffle(),
                ModelGd = new MultipleLinearRegression<LinearData>(),
                Optimizer = new Momentum(1E-1),
                Loss = new MeanSquared(),
                TrainPlan = new TrainPlan {Epoch = 50, BatchSize = 25},
                Metrics = new ObservableCollection<Metric>
                {
                    new ExplainedVariance(),
                    new LogCoshError(),
                    new MeanAbsoluteError(),
                    new MeanAbsolutePercentageError(),
                    new MeanRelativeError(),
                    new MeanSquaredError(),
                    new RSquared()
                },
                Print = _testOutputHelper.WriteLine
            };

            await trainer.Fit();
            print(trainer.ModelGd);
        }
    }
}