using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using ML.Core.Data;
using ML.Core.Data.Loader;
using ML.Core.Metrics;
using ML.Core.Metrics.Categorical;
using ML.Core.Models;
using ML.Core.Optimizers;
using ML.Core.Test.DataStructs;
using ML.Core.Trainers;
using Xunit;
using Xunit.Abstractions;
using CategoricalCrossentropy = ML.Core.Losses.CategoricalLosses.CategoricalCrossentropy;

namespace ML.Core.Test.OptdigitTest
{
    public class OptdigitTest : AbstractTest
    {
        private readonly string dataFolder = @"..\..\..\..\..\data";

        public OptdigitTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestLoadData()
        {
            var path = Path.Combine(dataFolder, "optdigits-train.csv");
            var dataset = TextLoader<OptdigitOneHot>.LoadDataSet(path, splitChar: ',');
            print(dataset);
        }

        [Fact]
        public void TestToOneHot()
        {
            var path = Path.Combine(dataFolder, "optdigits-train.csv");
            var dataset = TextLoader<OptdigitOneHot>.LoadDataSet(path, splitChar: ',');
            print(dataset.ToDatasetNDarray().Label);
        }

        private Dataset<DataView> GetOptdigitOnehot(string filename)
        {
            var trainpath = Path.Combine(dataFolder, filename);
            var dataset = TextLoader<OptdigitOneHot>.LoadDataSet(trainpath);
            return dataset;
        }

        [Fact]
        public async Task TestPerceptron()
        {
            var trainDataset = GetOptdigitOnehot("optdigits-train.csv");
            var valDataset = GetOptdigitOnehot("optdigits-val.csv");

            var trainer = new GDTrainer
            {
                TrainDataset = trainDataset.Shuffle(),
                ValDataset = valDataset.Shuffle(),
                ModelGd = new Perceptron(10),
                Optimizer = new Nadam(1E-2),
                Loss = new CategoricalCrossentropy(),

                TrainPlan = new TrainPlan {Epoch = 10, BatchSize = 50},
                Metrics = new ObservableCollection<Metric>
                {
                    new CategoricalAccuracy()
                },

                Print = _testOutputHelper.WriteLine
            };

            await trainer.Fit();
            print(trainer.ModelGd);
        }
    }
}