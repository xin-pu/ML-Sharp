using Numpy;

namespace ML.Core.Optimizers
{
    public class Adam : Optimizer
    {
        public Adam(double workLearningRate = 0.001, double beta1 = 0.9, double beta2 = 0.99) : base(workLearningRate)
        {
            Beta1 = beta1;
            Beta2 = beta2;
        }

        /// <summary>
        ///     M 衰减率
        /// </summary>
        public double Beta1 { protected set; get; }

        /// <summary>
        ///     G 衰减率
        /// </summary>
        public double Beta2 { protected set; get; }

        /// <summary>
        ///     梯度平方的指数加权平均
        /// </summary>
        public NDarray M { protected set; get; }

        /// <summary>
        ///     梯度的指数加权平均
        /// </summary>
        public NDarray G { protected set; get; }


        internal override NDarray call(NDarray weight, NDarray grad, int epoch)
        {
            if (epoch == 0)
            {
                M = np.zeros_like(weight);
                G = np.zeros_like(weight);
            }

            M = Beta1 * M + (1 - Beta1) * grad;
            G = Beta2 * G + (1 - Beta2) * np.square(grad);

            var m = M / (1 - Beta1);
            var g = G / (1 - Beta2);

            ///参数更新差值
            var delta_weight = -WorkLearningRate / np.sqrt(g + epsilon) * m;

            return weight + delta_weight;
        }
    }
}