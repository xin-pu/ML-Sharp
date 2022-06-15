using System.Collections.Generic;
using System.Linq;
using Numpy;
using Numpy.Models;

namespace ML.Core.Models
{
    public class KMeans : Cluster
    {
        private int _iterationLimit;

        public KMeans(int k, int iterationLimit = 100)
            : base(k)
        {
            IterationLimit = iterationLimit;
        }

        public int IterationLimit
        {
            get => _iterationLimit;
            set => Set(ref _iterationLimit, value);
        }

        public override NDarray Call(NDarray input)
        {
            var batchsize = input.shape[0];
            var centroidGroup = np.zeros(new Shape(batchsize), np.int32);
            var kmeans = np.array(Enumerable.Range(0, K)
                .Select(_ => input[np.random.choice(batchsize)]));

            var index = 0;
            while (index++ < IterationLimit)
            {
                var tempCentroidGroup = getCentroid(input, kmeans);

                var delta = tempCentroidGroup - centroidGroup;

                if (delta.GetData<int>().All(a => a == 0)) break;
                /// Update Kmeans
                kmeans = getKMeans(input, tempCentroidGroup);
                centroidGroup = tempCentroidGroup;
            }

            return centroidGroup;
        }


        /// <summary>
        ///     计算每个样本最近中心点序号
        /// </summary>
        /// <param name="input"></param>
        /// <param name="kmeans"></param>
        /// <returns></returns>
        internal NDarray getCentroid(NDarray input, NDarray kmeans)
        {
            var num_pts = input.shape[0];
            var num_feats = input.shape[1];

            var centroid_matrix = np.reshape(np.tile(kmeans, np.array(num_pts, 1)),
                new Shape(num_pts, K, num_feats));
            var point_matrix = np.reshape(np.tile(input, np.array(1, K)), new Shape(num_pts, K, num_feats));
            var dis = np.linalg.norm(point_matrix - centroid_matrix, axis: -1, ord: 2);
            var centroid_group = np.argmin(dis, -1).astype(np.int32);
            return centroid_group;
        }

        /// <summary>
        ///     计算按照分组，计算新的中心点
        /// </summary>
        /// <param name="input">所有样本</param>
        /// <param name="cluster">[batch size],每个样本当前最近中心点序号</param>
        /// <returns></returns>
        internal NDarray getKMeans(NDarray input, NDarray cluster)
        {
            var array = cluster.GetData<int>();
            var batch = cluster.shape[0];

            var dict = new Dictionary<int, List<NDarray>>();
            foreach (var i in Enumerable.Range(0, K))
            {
                var list = new List<NDarray>();
                foreach (var b in Enumerable.Range(0, batch))
                    if (array[b] == i)
                        list.Add(input[b]);
                dict[i] = list;
            }

            var kmeans = dict
                .Select(a =>
                {
                    var arr = np.vstack(a.Value.ToArray());
                    return np.average(arr, 0).astype(np.@double);
                }).ToArray();
            kmeans = kmeans.OrderBy(n => n.GetData<double>()[0]).ToArray();
            return np.vstack(kmeans);
        }
    }
}