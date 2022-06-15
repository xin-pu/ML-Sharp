using System.IO;
using ML.Core.Data.DataStructs;
using ML.Core.Data.Loader;
using ML.Core.Models;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test
{
    public class KMeansTest : AbstractTest
    {
        private readonly string dataFolder = @"..\..\..\..\..\data";

        public KMeansTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        [Fact]
        public void TestKmeans()
        {
            var path = Path.Combine(dataFolder, "data_cluster.txt");
            var data = TextLoader.LoadDataSet<LinearData>(path, new[] {','}, false);
            var input = data.ToFeatureNDarray();

            var kmeans = new KMeans(3);
            var res = kmeans.Call(input);
            print(res);
        }
    }
}