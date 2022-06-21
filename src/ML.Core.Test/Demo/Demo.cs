using System.IO;
using ML.Core.Data.DataStructs;
using ML.Core.Data.Loader;
using ML.Core.Transform;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test.Demo
{
    public class Demo : AbstractTest
    {
        private readonly string dataFolder = @"..\..\..\..\..\data";

        public Demo(ITestOutputHelper testOutputHelper) : base(testOutputHelper)
        {
        }

        [Fact]
        public void LoadDataSet()
        {
            var path = Path.Combine(dataFolder, "data_singlevar.txt");
            var Dataset = TextLoader.LoadDataSet<LinearData>(path, ',', false);
            print(Dataset);
        }

        [Fact]
        public void TestLeastSquare()
        {
            var path = Path.Combine(dataFolder, "data_singlevar.txt");
            var Dataset = TextLoader.LoadDataSet<LinearData>(path, ',', false);
            var datasetArray = Dataset.ToDatasetNDarray();
            var X = new LinearFirstorder().Call(datasetArray.Feature);
            var Y = datasetArray.Label;
            print(X);

            var weight = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y);
            print(weight);

            var res = np.linalg.lstsq(X, Y);
            print(res);
        }
    }
}