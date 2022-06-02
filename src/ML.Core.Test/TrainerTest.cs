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
        public async Task TestTrainer()
        {
            var path = Path.Combine(dataFolder, "iris-train.txt");


            var trainer = new Trainer<IrisData>
            {
                Model = new MultipleLinearRegression<IrisData>(),
                Dataset = TextLoader<IrisData>.LoadDataSet(path, splitChar: '\t'),
                Optimizer = new AdaDelta(),
                Loss = new MeanSquaredError()
            };

            var trainPlan = new TrainPlan
            {
                BatchSize = 4,
                Epoch = 100
            };

            await trainer.Fit(trainPlan);
        }
    }
}