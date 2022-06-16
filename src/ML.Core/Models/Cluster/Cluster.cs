using GalaSoft.MvvmLight;
using Numpy;

namespace ML.Core.Models
{
    public abstract class Cluster : ViewModelBase
    {
        private int _k;

        /// <summary>
        ///     聚类算法抽象类
        /// </summary>
        /// <param name="k"></param>
        protected Cluster(int k)
        {
            K = k;
        }

        public int K
        {
            get => _k;
            set => Set(ref _k, value);
        }

        public abstract NDarray Call(NDarray input);

        public static NDarray GetAdjacentMatrixEpsilon(double epsilon)
        {
            return null;
        }
    }
}