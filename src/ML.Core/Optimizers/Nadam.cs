using System.ComponentModel;
using Numpy;

namespace ML.Core.Optimizers
{
    public class Nadam : Optimizer
    {
        private double _beta1 = 0.9;
        private double _beta2 = 0.99;

        private NDarray _g;
        private NDarray _m;

        /// <summary>
        ///     Adaptive Moment Estimation Algorithm
        ///     Momentum  + RMSProp
        /// </summary>
        public Nadam()
        {
        }

        /// <summary>
        ///     Adaptive Moment Estimation Algorithm
        ///     Momentum  + RMSProp
        /// </summary>
        /// <param name="learningrate"></param>
        public Nadam(
            double learningrate)
            : base(learningrate)
        {
        }

        /// <summary>
        ///     M 衰减率
        /// </summary>
        [Category("Configuration")]
        public double Beta1
        {
            set => SetProperty(ref _beta1, value);
            get => _beta1;
        }

        /// <summary>
        ///     G 衰减率
        /// </summary>
        [Category("Configuration")]
        public double Beta2
        {
            set => SetProperty(ref _beta2, value);
            get => _beta2;
        }

        /// <summary>
        ///     梯度平方的指数加权平均
        /// </summary>
        [Category("State")]
        public NDarray M
        {
            protected set => SetProperty(ref _m, value);
            get => _m;
        }

        /// <summary>
        ///     梯度的指数加权平均
        /// </summary>
        [Category("State")]
        public NDarray G
        {
            protected set => SetProperty(ref _g, value);
            get => _g;
        }

        private NDarray m => M / (1 - Beta1);
        private NDarray g => G / (1 - Beta1);


        public override void Dispose()
        {
            M.Dispose();
            G.Dispose();
        }

        internal override NDarray call(NDarray weight, int epoch)
        {
            if (epoch == 0)
            {
                M = np.zeros_like(weight);
                G = np.zeros_like(weight);
            }


            var theda = weight - WorkLearningRate * m / (g + epsilon).sqrt();
            var grad = CalGradient(theda);

            M = Beta1 * M + (1 - Beta1) * grad;
            G = Beta2 * G + (1 - Beta2) * grad.square();

            ///参数更新差值
            var delta_weight = -WorkLearningRate * m / (g + epsilon).sqrt();

            return weight + delta_weight;
        }
    }
}