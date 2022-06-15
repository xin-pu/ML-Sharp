using System.IO;
using ML.Core.Data.DataStructs;
using ML.Core.Data.Loader;
using ML.Core.Models;
using Numpy;
using Numpy.Models;
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

            var kmeans = new KMeans(3, kMeansAlgorithm: KMeansAlgorithm.KMeansExt);
            var res = kmeans.Call(input);
            print(res);
        }

        [Fact]
        public void Test()
        {
            NDarray a = np.array(new double[,] {{1, 2, 1}, {2, 3, 2}, {3, 4, 4}, {5, 6, 5}});
            NDarray b = np.array(new double[,] {{2, 2, 3}});

            print(np.vstack(a, b));

            var kmeansCount = b.shape[0];
            var features = b.shape[1];

            var input = np.reshape(np.tile(a, np.array(kmeansCount)), new Shape(a.shape[0], kmeansCount, features));
            print(input);

            var kmeans = np.reshape(np.tile(b, np.array(a.shape[0], 1)), new Shape(a.shape[0], kmeansCount, features));
            print(kmeans);

            var dis = input - kmeans;
            print(dis);

            var dd = np.linalg.norm(input - kmeans, axis: 1, ord: 2);
            print(dd);

            var sumDis = np.sum(dd, 1, keepdims: true);
            print(sumDis);

            var max = np.argmax(sumDis, 0);
            print(max);
        }
    }
}