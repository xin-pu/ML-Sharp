using System.IO;
using System.Threading.Tasks;
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
        public async Task TestTrainer()
        {
            var path = Path.Combine(dataFolder, "data_singlevar.txt");

            var trainer = new Trainer<LinearData>
            {
                Model = new MultipleLinearRegression<LinearData>(),
                Dataset = TextLoader<LinearData>.LoadDataSet(path, false),
                Optimizer = new Momentum(1E-2),
                Loss = new MeanSquaredError(),
                TrainPlan = new TrainPlan {Epoch = 100, BatchSize = 25},
                Print = _testOutputHelper.WriteLine
            };


            await trainer.Fit();

            print(trainer.Model);
        }
    }
}