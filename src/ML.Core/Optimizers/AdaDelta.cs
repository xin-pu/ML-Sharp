using Numpy;

namespace ML.Core.Optimizers
{
    public class AdaDelta : Optimizer
    {
        public AdaDelta(double beta = 0.9) : base(0)
        {
            Beta = beta;
        }

        /// <summary>
        ///     衰减率
        /// </summary>
        public double Beta { protected set; get; }

        /// <summary>
        ///     参数梯度平方的累计值
        /// </summary>
        public NDarray G { protected set; get; }

        /// <summary>
        ///     参数更新差值δθ的平方的指数衰减权移动平均
        /// </summary>
        public NDarray Χ { protected set; get; }


        internal override NDarray call(NDarray weight, int epoch)
        {
            if (epoch == 0)
            {
                G = np.zeros_like(weight);
                Χ = np.zeros_like(weight);
            }

            var grad = CalGradient(weight);
            G = Beta * G + (1 - Beta) * np.square(grad);

            var deltaGrad = np.multiply(np.sqrt((Χ + epsilon) / (G + epsilon)), grad);

            Χ = Beta * Χ + (1 - Beta) * np.square(deltaGrad);

            return weight - deltaGrad;
        }
    }
}