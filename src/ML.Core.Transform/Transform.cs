using MvvmCross.ViewModels;
using Numpy;

namespace ML.Core.Transform
{
    /// <summary>
    ///     特征变换
    /// </summary>
    public abstract class Transform : MvxViewModel
    {
        public string Name => GetType().Name;

        /// <summary>
        ///     执行变换
        /// </summary>
        /// <param name="inputNDarray"></param>
        /// <returns></returns>
        public abstract NDarray Call(NDarray inputNDarray);

        public override string ToString()
        {
            return Name;
        }
    }
}