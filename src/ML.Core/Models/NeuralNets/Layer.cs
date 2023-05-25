using CommunityToolkit.Mvvm.ComponentModel;
using ML.Core.Optimizers;
using Numpy;

namespace ML.Core.Models.NeuralNets
{
    /// <typeparam name="T"></typeparam>
    public abstract class Layer : ObservableObject
    {
        public NDarray Output { set; get; } = np.empty();

        public abstract NDarray Forward(NDarray input);

        /// <summary>
        ///     更新当前权重```````````````````````````````````````````````````````````````````，返回梯度项
        /// </summary>
        /// <param name="gradient"></param>
        /// <param name="optimizer"></param>
        /// <returns></returns>
        public abstract NDarray Backward(NDarray gradient, Optimizer optimizer, int epoch = 0);
    }
}