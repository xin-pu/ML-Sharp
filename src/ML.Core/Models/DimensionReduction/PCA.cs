using System.Linq;
using GalaSoft.MvvmLight;
using Numpy;

namespace ML.Core.Models
{
    public class PCA : ViewModelBase
    {
        private int _dims;

        /// <summary>
        ///     主成成分分析
        /// </summary>
        /// <param name="dims"></param>
        public PCA(int dims)
        {
            Dims = dims;
        }

        public int Dims
        {
            get => _dims;
            set => Set(ref _dims, value);
        }

        public NDarray Weights { set; get; }

        public NDarray Call(NDarray sample)
        {
            var ave = np.average(sample, 0);

            var X = sample - ave;

            var (Lamda, V) = np.linalg.eig(np.dot(X.T, X));


            var v = np.argsort(Lamda).GetData<long>()
                .Select((index, dim) => (index, dim))
                .OrderByDescending(p => p.index)
                .Take(Dims)
                .Select(p => V.T[p.dim])
                .ToArray();

            var M = np.vstack(v).T;
            Weights = M;
            return M;
        }

        /// <summary>
        ///     按样本得到的投影向量，投影，将数据降维
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public NDarray Transform(NDarray input)
        {
            return np.dot(input, Weights);
        }
    }
}