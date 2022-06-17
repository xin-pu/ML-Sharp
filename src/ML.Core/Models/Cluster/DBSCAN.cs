using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.Random;
using Numpy;

namespace ML.Core.Models
{
    public class DBSCAN : Cluster
    {
        private double _epsilon;
        private int _minPoints;

        /// <summary>
        ///     密度聚类 DBSCAN
        /// </summary>
        /// <param name="k"></param>
        public DBSCAN(double epsilon, int minPoints)
        {
            Epsilon = epsilon;
            MinPoints = minPoints;
        }

        public double Epsilon
        {
            get => _epsilon;
            set => Set(ref _epsilon, value);
        }

        public int MinPoints
        {
            get => _minPoints;
            set => Set(ref _minPoints, value);
        }

        public override NDarray Call(NDarray input)
        {
            var batch = input.shape[0];
            var coreObjects = new List<int>();
            var allObjects = Enumerable.Range(0, batch).ToList();

            /// 计算核心对象
            foreach (var i in allObjects)
            {
                /// 获取领域样本
                var ner = getDirectly(i, input, Epsilon);
                if (ner.Length > MinPoints)
                    coreObjects.Add(i);
            }


            var cluster = new Dictionary<int, int[]>();
            var k = 0;
            while (coreObjects.Count > 0)
            {
                var allTemp = new List<int>(allObjects);

                /// 随机选取一个核心对象O;
                var coreObject = coreObjects[SystemRandomSource.Default.Next(0, coreObjects.Count)];
                var Q = new Queue<int>();
                Q.Enqueue(coreObject);

                while (Q.Count > 0)
                {
                    var q = Q.Dequeue();
                    var neighbors = getDirectly(q, input, Epsilon);
                    if (neighbors.Length > MinPoints)
                    {
                        var delta = neighbors.Intersect(allObjects);
                        delta.ToList().ForEach(d =>
                        {
                            Q.Enqueue(d);
                            allObjects.Remove(d);
                        });
                    }
                }

                var a = allTemp.Except(allObjects).ToArray();

                cluster[k++] = a;

                coreObjects.RemoveAll(arr => a.Contains((int) (NDarray) arr));
            }

            cluster[k] = allObjects.ToArray();


            var all = cluster
                .SelectMany(d =>
                    d.Value.Select(v => (v, d.Key)))
                .OrderBy(a => a.v)
                .Select(p => p.Key)
                .ToArray();
            var clusterNDarr = np.array(all);
            return clusterNDarr;
        }

        private int[] getDirectly(int index, NDarray input, double e)
        {
            var x = input[index];
            var dis = np.linalg.norm(input - x, axis: -1, ord: 2).GetData<double>();
            return dis.Select((d, i) => (d, i)).Where(p => p.d < e && p.d != 0).Select(p => p.i).ToArray();
        }
    }
}