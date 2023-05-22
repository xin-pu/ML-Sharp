using System.ComponentModel;
using Numpy;

namespace ML.Core.Optimizers
{
    /// <summary>
    /// </summary>
    public class AdaGrad : Optimizer
    {
        private NDarray _g = np.empty();

        /// <summary>
        ///     自适应学习率
        ///     Adapter Gradient Algorithm
        ///     每次迭代时自适应地调整每个参数的学习率
        ///     初始学习率1E-3
        /// </summary>
        public AdaGrad()
        {
        }


        /// <summary>
        ///     自适应学习率
        ///     Adapter Gradient Algorithm
        ///     每次迭代时自适应地调整每个参数的学习率
        /// </summary>
        /// <param name="learningrate"></param>
        public AdaGrad(double learningrate)
            : base(learningrate)
        {
        }

        /// <summary>
        ///     参数梯度平方的累计值
        /// </summary>
        [Category("State")]
        public NDarray G
        {
            protected set => SetProperty(ref _g, value);
            get => _g;
        }

        public override void Dispose()
        {
            G.Dispose();
        }

        internal override NDarray call(NDarray weight, int epoch)
        {
            var grad = CalGradient(weight);

            if (epoch == 0)
                G = np.zeros_like(weight);

            G += grad.square();
            var delta = -(WorkLearningRate / (G + epsilon).sqrt()).multiply(grad);
            return weight + delta;
        }
    }
}