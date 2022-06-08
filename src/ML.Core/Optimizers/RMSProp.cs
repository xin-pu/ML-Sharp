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
        /// <param name="workLearningRate">初始学习率</param>
        public RMSProp(double workLearningRate)
            : base(workLearningRate)
        {
        }


        /// <summary>
        ///     衰减率
        /// </summary>
        public double Beta
        {
            set => SetProperty(ref _beta, value);
            get => _beta;
        }

        /// <summary>
        ///     参数梯度平方的累计值
        /// </summary>
        public NDarray G
        {
            protected set => SetProperty(ref _g, value);
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