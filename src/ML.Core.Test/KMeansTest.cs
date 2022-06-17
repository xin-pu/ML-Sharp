using System.IO;
using ML.Core.Data;
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
        public void TestKMeans()
        {
            var path = Path.Combine(dataFolder, "data_cluster.txt");
            var data = TextLoader.LoadDataSet<LinearData>(path, new[] {','}, false);

            var kmeans = new KMeans(4, kMeansAlgorithm: KMeansAlgorithm.KMeans);
            var res = kmeans.Call(data.Value);

            foreach (var p in res)
            {
                var array = new Dataset<DataView>(p.Value).ToFeatureNDarray();
                print($"{p.Key}:\t{p.Value.Length}\r\n{array}\r\n{new string('-', 30)}");
            }
        }


        [Fact]
        public void TestKMeansExt()
        {
            var path = Path.Combine(dataFolder, "data_cluster.txt");
            var data = TextLoader.LoadDataSet<LinearData>(path, new[] {','}, false);

            var kmeans = new KMeans(4, kMeansAlgorithm: KMeansAlgorithm.KMeansExt);
            var res = kmeans.Call(data.Value);
            foreach (var p in res)
            {
                var array = new Dataset<DataView>(p.Value).ToFeatureNDarray();
                print($"{p.Key}:\t{p.Value.Length}\r\n{array}\r\n{new string('-', 30)}");
            }
        }


        [Fact]
        public void TestSpectral()
        {
            var path = Path.Combine(dataFolder, "data_cluster.txt");
            var data = TextLoader.LoadDataSet<LinearData>(path, new[] {','}, false);


            var spectral = new Spectral(4);
            var res = spectral.Call(data.Value);
            foreach (var p in res)
            {
                var array = new Dataset<DataView>(p.Value).ToFeatureNDarray();
                print($"{p.Key}:\t{p.Value.Length}\r\n{array}\r\n{new string('-', 30)}");
            }
        }


        [Fact]
        public void TestDBSCAN()
        {
            var path = Path.Combine(dataFolder, "data_cluster.txt");
            var data = TextLoader.LoadDataSet<LinearData>(path, new[] {','}, false);


            var spectral = new DBSCAN(0.95, 10);
            var res = spectral.Call(data.Value);
            foreach (var p in res)
            {
                var array = new Dataset<DataView>(p.Value).ToFeatureNDarray();
                print($"{p.Key}:\t{p.Value.Length}\r\n{array}\r\n{new string('-', 30)}");
            }
        }
    }
}