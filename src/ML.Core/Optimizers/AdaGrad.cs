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

        public override NDarray Call(NDarray weight, NDarray gradient, int epoch)
        {
            if (epoch == 0)
                G = np.zeros_like(weight);

            G += gradient.square();
            var delta = -(WorkLearningRate / (G + epsilon).sqrt()).multiply(gradient);
            return weight + delta;
        }
    }
}