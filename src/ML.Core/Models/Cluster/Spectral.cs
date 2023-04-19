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
        private int _k;
        private LaprasMatrixType _laprasMatrixType;

        /// <summary>
        ///     谱聚类
        /// </summary>
        /// <param name="k"></param>
        public Spectral(int k, LaprasMatrixType laprasMatrixType = LaprasMatrixType.Lsym)

        {
            K = k;
            LaprasMatrixType = laprasMatrixType;
        }

        public int K
        {
            get => _k;
            set => Set(ref _k, value);
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
            var v = Lamda.argsort().GetData<long>()
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
            var D = np.eye(input.shape[0]) * W.sum(0);
            return (W, D);
        }

        internal NDarray getLaprasMatrix(NDarray D, NDarray W)
        {
            var L = D - W;
            switch (LaprasMatrixType)
            {
                /// Lsym 标准化 的拉普拉斯矩阵
                /// Lsym = D^-0.5*L*D^-0.5 = I- D^-0.5*W*D^-0.5
                case LaprasMatrixType.Lsym:
                    var ds = np.linalg.inv(D.power(np.array(0.5)));
                    return ds.dot(L).dot(ds);

                /// Lrw 标准化 的拉普拉斯矩阵
                /// Lrw = D^-1*L = I- D^-1*W
                case LaprasMatrixType.Lrw:
                    return np.linalg.inv(D).dot(L);

                /// 非标准化的拉普拉斯矩阵
                default:
                    return L;
            }
        }
    }
}