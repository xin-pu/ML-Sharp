using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.Random;
using ML.Core.Data;
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
        public void TestDe()
        {
            var path = Path.Combine(dataFolder, "data_cluster.txt");
            var data = TextLoader.LoadDataSet<LinearData>(path, new[] {','}, false);
            var input = data.ToFeatureNDarray();
            var batch = input.shape[0];


            var epsilon = 1;
            var minPoints = 20;
            var coreObjects = new List<NDarray>();
            var allObjects = new List<NDarray>();

            /// 计算核心对象
            foreach (var i in Enumerable.Range(0, batch))
            {
                var x = input[i];
                allObjects.Add(x);
                /// 获取领域样本
                var ner = getDirectly(x, input, epsilon);
                if (ner.Length > minPoints)
                    coreObjects.Add(x);
            }

            var cluster = new List<NDarray[]>();

            while (coreObjects.Count > 0)
            {
                var allTemp = new List<NDarray>(allObjects);

                /// 随机选取一个核心对象O;
                var coreObject = coreObjects[SystemRandomSource.Default.Next(0, coreObjects.Count)];
                allObjects.Remove(coreObject);
                var Q = new Queue<NDarray>();
                Q.Enqueue(coreObject);


                while (Q.Count > 0)
                {
                    var q = Q.Dequeue();
                    var neighbors = getDirectly(q, input, epsilon);
                    if (neighbors.Length > minPoints)
                    {
                        var delta = neighbors
                            .Where(arr => allObjects.Contains(arr) && !Equals(arr, q))
                            .ToArray();
                        delta.ToList().ForEach(d =>
                        {
                            Q.Enqueue(d);
                            allObjects.Remove(d);
                        });
                    }
                }

                var a = allTemp.Where(arr => !allObjects.Contains(arr)).ToList();

                cluster.Add(a.ToArray());

                coreObjects.RemoveAll(arr => a.Contains(arr));
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