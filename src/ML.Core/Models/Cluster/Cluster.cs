using CommunityToolkit.Mvvm.ComponentModel;
using ML.Core.Data;
using Numpy;

namespace ML.Core.Models
{
    public abstract class Cluster : ObservableObject
    {
        /// <summary>
        ///     聚类算法抽象类
        /// </summary>
        /// <param name="k"></param>
        protected Cluster()
        {
        }


        /// <summary>
        ///     返回Input数据的分类数组
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public abstract NDarray Call(NDarray input);

        /// <summary>
        ///     输入N个DataView的集合
        ///     聚类成K个DataView集合
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public virtual Dictionary<int, DataView[]> Call(DataView[] input)
        {
            var inputArrays = input.Select(a => a.ToFeatureNDarray()).ToArray();
            var inputNDArray = np.vstack(inputArrays);

            var centroidGroup = Call(inputNDArray);

            var pair = centroidGroup.GetData<int>()
                .Select((n, i) => (n, i))
                .GroupBy(p => p.n)
                .ToDictionary(
                    p => p.Key,
                    p => p.Select(c => input[c.i]).ToArray());

            return pair;
        }
    }
}