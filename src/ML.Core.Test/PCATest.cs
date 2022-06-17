using System.IO;
using ML.Core.Data.DataStructs;
using ML.Core.Data.Loader;
using ML.Core.Models;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test
{
    public class PCATest : AbstractTest
    {
        private readonly string dataFolder = @"..\..\..\..\..\data";

        public PCATest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestPCA()
        {
            var path = Path.Combine(dataFolder, "iris-train.txt");
            var data = TextLoader.LoadDataSet<IrisData>(path, new[] {'\t'});
            var sample = data.ToDatasetNDarray().Feature;


            var pca = new PCA(2);
            pca.Call(sample);
            print(pca.Transform(sample));

            path = Path.Combine(dataFolder, "iris-test.txt");
            data = TextLoader.LoadDataSet<IrisData>(path, new[] {'\t'});
            sample = data.ToDatasetNDarray().Feature;

            var newSample = pca.Transform(sample);
            print(newSample);
        }
    }
}