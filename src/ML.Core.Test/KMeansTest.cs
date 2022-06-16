using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.Random;
using ML.Core.Data.DataStructs;
using ML.Core.Data.Loader;
using ML.Core.Models;
using Numpy;
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
        public void TestDe()
        {
            var path = Path.Combine(dataFolder, "data_cluster.txt");
            var data = TextLoader.LoadDataSet<LinearData>(path, new[] {','}, false);
            var input = data.ToFeatureNDarray();

            var batch = input.shape[0];
            var e = 1.5;
            var minPoints = 40;
            var ou = new List<NDarray>();
            var all = new List<NDarray>();

            /// 计算核心对象
            foreach (var i in Enumerable.Range(0, batch))
            {
                var x = input[i];
                all.Add(x);
                var ner = getDirectly(x, input, e);
                if (ner.Length > minPoints)
                    ou.Add(x);
            }


            while (ou.Count > 0)
            {
                var allTemp = new List<NDarray>(all);

                ///随机选取一个核心对象O;
                var o = ou[SystemRandomSource.Default.Next(0, ou.Count)];
                all.Remove(o);
                var Q = new Queue<NDarray>();
                Q.Enqueue(o);


                while (Q.Count > 0)
                {
                    var q = Q.Dequeue();
                    var n = getDirectly(q, input, e);
                    if (n.Length > minPoints)
                    {
                        var delta = n.Where(arr => all.Contains(arr) && !Equals(arr, q)).ToArray();
                        delta.ToList().ForEach(d =>
                        {
                            Q.Enqueue(d);
                            all.Remove(d);
                        });
                    }
                }

                var a = allTemp.Where(arr => !all.Contains(arr)).ToList();
                print(np.vstack(a.ToArray()));

                ou.RemoveAll(arr => a.Contains(arr));
            }
        }

        private NDarray[] getDirectly(NDarray x, NDarray input, double e)
        {
            var dis = np.linalg.norm(input - x, axis: -1, ord: 2).GetData<double>();
            return dis
                .Select((d, i) => (d, i))
                .Where(p => p.d < e && p.d != 0)
                .Select(p => input[p.i])
                .ToArray();
        }
    }
}