using System.Text;
using CommunityToolkit.Mvvm.ComponentModel;
using Numpy;

namespace ML.Core.Transform
{
    /// <summary>
    ///     特征变换
    /// </summary>
    public abstract class Transformer : ObservableObject
    {
        public string Name => GetType().Name;

        public abstract bool IsKernel { get; }

        /// <summary>
        ///     执行变换
        /// </summary>
        /// <param name="inputNDarray"></param>
        /// <returns></returns>
        public abstract NDarray Call(NDarray inputNDarray);

        public override string ToString()
        {
            var str = new StringBuilder();
            str.Append($"Transform:\t{Name}");
            return str.ToString();
        }
    }
}