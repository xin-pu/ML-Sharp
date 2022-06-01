using Numpy;

namespace ML.Core.Optimizers
{
    /// <summary>
    ///     variables=variables-η*f'(variables)
    ///     一维梯度下降
    /// </summary>
    public class SGD : Optimizer
    {
        public SGD(double workLearningRate) : base(workLearningRate)
        {
        }

        internal override NDarray call(NDarray weight, NDarray grad, int epoch)
        {
            var delta = -grad * WorkLearningRate;
            return weight + delta;
        }
    }
}