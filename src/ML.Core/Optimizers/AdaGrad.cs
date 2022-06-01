using Numpy;

namespace ML.Core.Optimizers
{
    /// <summary>
    /// </summary>
    public class AdaGrad : Optimizer
    {
        /// <summary>
        ///     自适应学习率
        ///     adapter gradient algorithm
        /// </summary>
        /// <param name="beta">学习率</param>
        public AdaGrad(double workLearningRate)
            : base(workLearningRate)
        {
        }

        /// <summary>
        ///     参数梯度平方的累计值
        /// </summary>
        public NDarray G { protected set; get; }

        internal override NDarray call(NDarray weight, NDarray grad, int epoch)
        {
            if (epoch == 0)
                G = np.zeros_like(weight);

            G += np.square(grad);
            var delta = -np.multiply(WorkLearningRate / np.sqrt(G + epsilon), grad);
            return weight + delta;
        }
    }
}