using System.ComponentModel;
using Numpy;

namespace ML.Core.Optimizers
{
    public class RMSProp : Optimizer
    {
        private double _beta = 0.9;
        private NDarray _g;

        /// <summary>
        ///     自适应学习率
        ///     初始学习率，默认1E-3, Beta=0.9
        /// </summary>
        public RMSProp()
        {
        }

        /// <summary>
        ///     自适应学习率
        /// </summary>
        /// <param name="learningrate">初始学习率</param>
        public RMSProp(double learningrate)
            : base(learningrate)
        {
        }


        /// <summary>
        ///     衰减率
        /// </summary>
        [Category("Configuration")]
        public double Beta
        {
            set => Set(ref _beta, value);
            get => _beta;
        }

        /// <summary>
        ///     参数梯度平方的累计值
        /// </summary>
        [Category("State")]
        public NDarray G
        {
            protected set => Set(ref _g, value);
            get => _g;
        }

        internal override NDarray call(NDarray weight, int epoch)
        {
            if (epoch == 0)
                G = np.zeros_like(weight);

            var grad = CalGradient(weight);
            G = Beta * G + (1 - Beta) * np.square(grad);
            var delta = -np.multiply(WorkLearningRate / np.sqrt(G + epsilon), grad);
            return weight + delta;
        }
    }
}