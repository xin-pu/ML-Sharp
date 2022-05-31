using AutoDiff;
using ML.Core.Data;
using ML.Core.Transform;
using MvvmCross.ViewModels;
using Numpy;

namespace ML.Core.Models
{
    /// <summary>
    ///     基本模型抽象类
    ///     需要定义转换器，默认无转换
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class Model : MvxViewModel
    {
        public string Name => GetType().Name;

        public Transformer Transformer { set; get; }
    }

    /// <summary>
    ///     用于梯度下降法的基本模型
    ///     需要定义损失函数
    ///     需要定义优化器
    ///     需要定义转换器
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class GDModel<T> : Model
        where T : DataView
    {
        public Variable[] Variables { set; get; }

        public abstract Term CallGraph(NDarray x);

        public abstract NDarray Call(NDarray x);
    }
}