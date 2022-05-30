using System.Security.Cryptography;
using ML.Core.Data;
using MvvmCross.ViewModels;

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

        public Transform.Transformer 
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
    }
}