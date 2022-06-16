using System.Linq;
using ML.Core.Transform;
using Numpy;

namespace ML.Core.Models
{
    public enum LaprasMatrixType
    {
        L,
        Lsym,
        Lrw
    }

    public class Spectral : Cluster
    {
        private LaprasMatrixType _laprasMatrixType;

        /// <summary>
        ///     谱聚类
        /// </summary>
        /// <param name="k"></param>
        public Spectral(int k, LaprasMatrixType laprasMatrixType = LaprasMatrixType.Lsym)
            : base(k)
        {
            LaprasMatrixType = laprasMatrixType;
        }

        public LaprasMatrixType LaprasMatrixType
        {
            get => _laprasMatrixType;
            set => Set(ref _laprasMatrixType, value);
        }

        public override NDarray Call(NDarray input)
        {
            ///计算邻接矩阵W和度举证D
            var (W, D) = getAdjacentMatrix(input);

            /// 计算非标准化的拉普拉斯特征矩阵
            var laprasMatrix = getLaprasMatrix(D, W);

            var (Lamda, V) = np.linalg.eig(laprasMatrix);
            /// 计算特征值和特征向量

            ///将K的最小特征值的特征向量组合成n*K维矩阵M
            var v = np.argsort(Lamda).GetData<long>()
                .Select((index, dim) => (index, dim))
                .OrderBy(p => p.index)
                .Take(K)
                .Select(p => V.T[p.dim])
                .ToArray();
            var M = np.vstack(v).T;

            /// 对M进行K均值聚类算法
            var kmeans = new KMeans(K, kMeansAlgorithm: KMeansAlgorithm.KMeansExt);
            var res = kmeans.Call(M);
            return res;
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

        internal NDarray getLaprasMatrix(NDarray D, NDarray W)
        {
            var L = D - W;
            switch (LaprasMatrixType)
            {
                case LaprasMatrixType.Lsym:
                    var ds = np.linalg.inv(np.power(D, np.array(0.5)));
                    return np.dot(np.dot(ds, L), ds);
                case LaprasMatrixType.Lrw:
                    return np.dot(np.linalg.inv(D), L);
                default:
                    return L;
            }
        }
    }
}