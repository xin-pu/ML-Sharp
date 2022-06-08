using Numpy;

namespace ML.Core.Optimizers
{
    public class Adam : Optimizer
    {
        private double _beta1 = 0.9;
        private double _beta2 = 0.99;

        private NDarray _g;
        private NDarray _m;

        /// <summary>
        ///     Adaptive Moment Estimation Algorithm
        ///     Momentum  + RMSProp
        /// </summary>
        public Adam()
        {
        }

        /// <summary>
        ///     Adaptive Moment Estimation Algorithm
        ///     Momentum  + RMSProp
        /// </summary>
        /// <param name="workLearningRate"></param>
        public Adam(double workLearningRate)
            : base(workLearningRate)
        {
        }


        /// <summary>
        ///     M 衰减率
        /// </summary>
        public double Beta1
        {
            set => Set(ref _beta1, value);
            get => _beta1;
        }

        /// <summary>
        ///     G 衰减率
        /// </summary>
        public double Beta2
        {
            set => Set(ref _beta2, value);
            get => _beta2;
        }

        /// <summary>
        ///     梯度平方的指数加权平均
        /// </summary>
        public NDarray M
        {
            protected set => Set(ref _m, value);
            get => _m;
        }

        /// <summary>
        ///     梯度的指数加权平均
        /// </summary>
        public NDarray G
        {
            protected set => Set(ref _g, value);
            get => _g;
        }


        internal override NDarray call(NDarray weight, int epoch)
        {
            if (epoch == 0)
            {
                M = np.zeros_like(weight);
                G = np.zeros_like(weight);
            }


            var grad = CalGradient(weight);

            M = Beta1 * M + (1 - Beta1) * grad;
            G = Beta2 * G + (1 - Beta2) * np.square(grad);

            var m = M / (1 - Beta1);
            var g = G / (1 - Beta2);

            ///参数更新差值
            var delta_weight = -WorkLearningRate * m / np.sqrt(g + epsilon);

            return weight + delta_weight;
        }
    }
}