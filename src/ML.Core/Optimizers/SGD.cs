using Numpy;

namespace ML.Core.Optimizers
{
    /// <summary>
    ///     一维梯度下降
    /// </summary>
    public class SGD : Optimizer
    {
        /// <summary>
        ///     随机梯度下降
        ///     variables=variables-η*f'(variables)
        ///     默认初始学习率1E-3
        /// </summary>
        public SGD()
        {
        }


        /// <summary>
        ///     随机梯度下降
        ///     variables=variables-η*f'(variables)
        /// </summary>
        /// <param name="workLearningRate">初始学习率</param>
        public SGD(double workLearningRate)
            : base(workLearningRate)
        {
        }

        internal override NDarray call(NDarray weight, int epoch)
        {
            var grad = CalGradient(weight);
            var delta = -grad * WorkLearningRate;
            return weight + delta;
        }
    }
}