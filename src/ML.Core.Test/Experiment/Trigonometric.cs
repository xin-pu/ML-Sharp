using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ML.Core.Data.DataStructs;
using ML.Core.Data.Loader;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Metrics.Regression;
using ML.Core.Models;
using ML.Core.Optimizers;
using ML.Core.Trainers;
using ML.Core.Transform;
using Numpy;
using Numpy.Models;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test.Experiment
{
    public class Trigonometric : AbstractTest
    {
        public Trigonometric(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        public double func(double x, double A, double B, double k, double b)
        {
            return A * Math.Sin(k * x + b) + B;
        }

        public Tuple<double[], double[]> CreateDemo(double[] parameters)
        {
            var x_list = Enumerable.Range(0, 500).Select(a => a * 0.01).ToArray();
            var y_list = x_list.Select(x => func(x, 0.5, 1, 0.5, 0.1)).ToArray();

            return new Tuple<double[], double[]>(x_list, y_list);
        }

        [Fact]
        public void CreateData()
        {
            var (x_list, y_list) = CreateDemo(new[] {0.5, 1, 0.5, 0.1});
            var (x_list1, y_list1) = CreateDemo(new[]
                {-0.49105918407440186, 1.0052903890609741, 0.5064752697944641, 3.232282876968384});
            var (x_list12, y_list2) = CreateDemo(new[]
                {-0.49403616786003113, 1.006188988685608, -0.5034294724464417, -0.08934494107961655});
            var s = x_list
                .Zip(y_list, (a, b) => $"{a}\t{b}")
                .Zip(y_list1, (a, b) => $"{a}\t{b}")
                .Zip(y_list2, (a, b) => $"{a}\t{b}")
                .ToArray();
            print(string.Join("\r\n", s));
        }

        [Fact]
        public void Test()
        {
            var x_list = Enumerable.Range(0, 1000).Select(a => a * 0.01).ToArray();
            var d = new TriangleRegression(1)
            {
                Weights = np.array(0.122017440864487, 1.4585780028113, 0.125968519175074)
            };
            var y_list = d.Call(np.array(x_list).reshape(new Shape(1000, 1)));
            print(string.Join("\r", y_list.GetData<double>()));
        }

        [Fact]
        public async Task TestPolyRegression()
        {
            var path = Path.Combine(@"E:\Code MachineLearning\ML-Sharp\data\CathyDemo.txt");

            var trainer = new GDTrainer
            {
                TrainDataset = TextLoader.LoadDataSet<LinearData>(path, '\t', false),
                ValDataset = TextLoader.LoadDataSet<LinearData>(path, '\t', false),
                ModelGd = new TriangleRegression(1),
                Optimizer = new Nadam(1E-3),
                Loss = new MeanSquared(),
                TrainPlan = new TrainPlan {Epoch = 100, BatchSize = 100},
                Metrics = new ObservableCollection<Metric>
                {
                    new MeanAbsoluteError()
                },
                Print = _testOutputHelper.WriteLine
            };

            await trainer.Fit();
            print(trainer.ModelGd);
        }

        [Fact]
        public void ByDot()
        {
            var path = Path.Combine(@"E:\Code MachineLearning\ML-Sharp\data\CathyDemo.txt");
            var TrainDataset = TextLoader.LoadDataSet<LinearData>(path, '\t', false);
            var c = TrainDataset.ToDatasetNDarray();


            var X = new TrianglePoly(1).Call(c.Feature);
            var Y = c.Label;


            var weight = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y);

            print(string.Join(",", weight.GetData<double>()));
            var res = np.linalg.lstsq(X, Y);
            print(res);
        }

        [Fact]
        public void T()
        {
            var path = Path.Combine(@"E:\Code MachineLearning\ML-Sharp\data\CathyDemo.txt");
            var TrainDataset = TextLoader.LoadDataSet<LinearData>(path, '\t', false);
            var c = TrainDataset.ToDatasetNDarray();

            var d = np.fft.fftfreq(1000, 0.01f);
            var dd = np.fft.fft_(c.Label);
            var ddd = d[dd["1:"].argmax() + 1];
        }
    }
}