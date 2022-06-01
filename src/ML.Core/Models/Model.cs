using AutoDiff;
using ML.Core.Data;
using ML.Core.Transform;
using Numpy;

namespace ML.Core.Models
{
    /// <summary>
    ///     用于梯度下降法的基本模型
    ///     需要定义损失函数
    ///     需要定义优化器
    ///     需要定义转换器
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class Model<T>
        where T : DataView
    {
        public string Name => GetType().Name;

        public Transformer Transformer { set; get; }

        public Variable[] Variables { set; get; }

        public abstract Term CallGraph(NDarray x);

        public abstract NDarray Call(NDarray x);
    }
}