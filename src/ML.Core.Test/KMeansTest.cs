using System.IO;
using ML.Core.Data.DataStructs;
using ML.Core.Data.Loader;
using ML.Core.Models;
using ML.Core.Transform;
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


        [Fact]
        public void TestAdjacentMatrix()
        {
            var input = np.array(new[,] {{1, 1, 1}, {1, 2, 1}, {2, 2.5, 2}, {3.1, 3, 3}});
            var (W, D) = getAdjacentMatrix(input);
            print(W);
            print(D);

            var Lrw = np.eye(input.shape[0]) - np.linalg.inv(D) * W;
            print(Lrw);
        }


        [Fact]
        public void TestAdjacentMatrix2()
        {
            var path = Path.Combine(dataFolder, "data_cluster.txt");
            var data = TextLoader.LoadDataSet<LinearData>(path, new[] {','}, false);
            var input = data.ToFeatureNDarray();
            var (W, D) = getAdjacentMatrix(input);
            print(W);
            print(D);

            var Lrw = np.eye(input.shape[0]) - np.linalg.inv(D) * W;
            print(Lrw);

            var (l, v) = np.linalg.eig(Lrw);
            print(l);
            print(v);
        }

        /// <summary>
        ///     epsilon
        /// </summary>
        /// <param name="input"></param>
        /// <param name="esplion"></param>
        private (NDarray, NDarray) getAdjacentEpsilon(NDarray input, double epsilon)
        {
            var batch = input.shape[0];
            var features = input.shape[1];
            var inputH = np.reshape(np.tile(input, np.array(batch)), new Shape(batch, batch, features));

            var inputV = np.reshape(np.tile(input, np.array(batch, 1)), new Shape(batch, batch, features));

            var dis = np.linalg.norm(inputV - inputH, axis: -1, ord: 2);

            var W = np.where(dis < epsilon, np.array(epsilon), np.array(0));
            //np.fill_diagonal(W, 0);

            var D = np.eye(batch) * np.sum(W, 0);

            return (W, D);
        }

        /// <summary>
        ///     epsilon
        /// </summary>
        /// <param name="input"></param>
        /// <param name="esplion"></param>
        private (NDarray, NDarray) getAdjacentMatrix(NDarray input)
        {
            var W = new Gaussian().Call(input);
            var D = np.eye(input.shape[0]) * np.sum(W, 0);
            return (W, D);
        }
    }
}