using Numpy;
using Numpy.Models;

namespace ML.Core.Models
{
    public class KMeans : Cluster
    {
        private int _iterationLimit;

        private int _k;
        private KMeansAlgorithm _kMeansAlgorithm;

        /// <summary>
        ///     KMeans 聚类算法
        ///     采用了贪心策略，通过迭代优化来近似去接
        /// </summary>
        /// <param name="k"></param>
        /// <param name="iterationLimit">limit for iteration</param>
        /// <param name="kMeansAlgorithm">KMeans and KMeans++ </param>
        public KMeans(
            int k,
            int iterationLimit = 100,
            KMeansAlgorithm kMeansAlgorithm = KMeansAlgorithm.KMeans)
        {
            K = k;
            IterationLimit = iterationLimit;
            KMeansAlgorithm = kMeansAlgorithm;
        }

        public int K
        {
            get => _k;
            set => SetProperty(ref _k, value);
        }

        public KMeansAlgorithm KMeansAlgorithm
        {
            get => _kMeansAlgorithm;
            set => SetProperty(ref _kMeansAlgorithm, value);
        }

        public int IterationLimit
        {
            get => _iterationLimit;
            set => SetProperty(ref _iterationLimit, value);
        }

        public override NDarray Call(NDarray inputNDArray)
        {
            var batchsize = inputNDArray.shape[0];
            var centroidGroup = np.zeros(new Shape(batchsize), np.int32);
            var kmeans = KMeansAlgorithm == KMeansAlgorithm.KMeansExt
                ? extendCentroid(inputNDArray)
                : randomCentroid(inputNDArray);

            var index = 0;
            while (index++ < IterationLimit)
            {
                var tempCentroidGroup = getCentroid(inputNDArray, kmeans);

                var delta = tempCentroidGroup - centroidGroup;

                if (delta.GetData<int>().All(a => a == 0)) break;
                /// Update Kmeans
                kmeans = getKMeans(inputNDArray, tempCentroidGroup);
                centroidGroup = tempCentroidGroup;
            }

            return centroidGroup;
        }


        public NDarray randomCentroid(NDarray input)
        {
            var batchsize = input.shape[0];
            var kmeans = np.array(Enumerable.Range(0, K)
                .Select(_ => input[np.random.choice(batchsize)]));
            return kmeans;
        }

        public NDarray extendCentroid(NDarray input)
        {
            var batch = input.shape[0];
            var features = input.shape[1];
            var kmeans = input[np.random.choice(batch)].expand_dims(0);

            foreach (var _ in Enumerable.Range(1, K - 1))
            {
                var kmeansCount = kmeans.shape[0];

                var inputTile = input.tile(np.array(kmeansCount)).reshape(new Shape(batch, kmeansCount, features));
                var kmeanTile = kmeans.tile(np.array(batch, 1)).reshape(new Shape(batch, kmeansCount, features));

                var dis = np.linalg.norm(inputTile - kmeanTile, axis: 1, ord: 2);
                var disSum = dis.sum(1, keepdims: true);
                var max = disSum.argmax(0).GetData<int>()[0];
                kmeans = np.vstack(kmeans.copy(), input[max]);
            }

            return kmeans;
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

            var centroid_matrix = kmeans.tile(np.array(num_pts, 1)).reshape(new Shape(num_pts, K, num_feats));
            var point_matrix = input.tile(np.array(1, K)).reshape(new Shape(num_pts, K, num_feats));
            var dis = np.linalg.norm(point_matrix - centroid_matrix, axis: -1, ord: 2);
            var centroid_group = dis.argmin(-1).astype(np.int32);
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
                    return arr.average(0).astype(np.@double);
                }).ToArray();
            kmeans = kmeans.OrderBy(n => n.GetData<double>()[0]).ToArray();
            return np.vstack(kmeans);
        }
    }

    public enum KMeansAlgorithm
    {
        KMeans,
        KMeansExt
    }
}