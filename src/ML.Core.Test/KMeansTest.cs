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
        public void TestKMeans()
        {
            var path = Path.Combine(dataFolder, "data_cluster.txt");
            var data = TextLoader.LoadDataSet<LinearData>(path, new[] {','}, false);
            var input = data.ToFeatureNDarray();

            var kmeans = new KMeans(3, kMeansAlgorithm: KMeansAlgorithm.KMeans);
            var res = kmeans.Call(input);
            print(res);
        }


        [Fact]
        public void TestKMeansExt()
        {
            var path = Path.Combine(dataFolder, "data_cluster.txt");
            var data = TextLoader.LoadDataSet<LinearData>(path, new[] {','}, false);
            var input = data.ToFeatureNDarray();

            var kmeans = new KMeans(3, kMeansAlgorithm: KMeansAlgorithm.KMeansExt);
            var res = kmeans.Call(input);
            print(res);
        }


        [Fact]
        public void TestSpectral()
        {
            var path = Path.Combine(dataFolder, "data_cluster.txt");
            var data = TextLoader.LoadDataSet<LinearData>(path, new[] {','}, false);
            var input = data.ToFeatureNDarray();

            var spectral = new Spectral(3);
            var res = spectral.Call(input);
            print(res);
        }

        [Fact]
        public void TestSpectral2()
        {
            var path = Path.Combine(dataFolder, "data_cluster.txt");
            var data = TextLoader.LoadDataSet<LinearData>(path, new[] {','}, false);
            var input = data.ToFeatureNDarray();

            var (W, D) = getAdjacentMatrix(input);
            print(W);
            print(D);
            print(np.eye(4));
            print(np.eye(input.shape[0]) - np.linalg.inv(D) * W);

            var Lrw = np.dot(np.linalg.inv(D), W);
            print(Lrw);


            var (Lamda, V) = np.linalg.eig(Lrw);
            print(Lamda);
            print(V);
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
        ///     全连接法计算邻接矩阵
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        internal (NDarray, NDarray) getAdjacentMatrix(NDarray input)
        {
            var W = new Gaussian().Call(input);
            var D = np.eye(input.shape[0]) * np.sum(W, 0);
            return (W, D);
        }
    }
}