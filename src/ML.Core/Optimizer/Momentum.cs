using Numpy;

namespace ML.Core.Optimizer
{
    public class Momentum : Optimizer
    {
        /// <summary>
        ///     动量法
        /// </summary>
        /// <param name="learningrate"></param>
        /// <param name="rho"></param>
        public Momentum(double learningrate, double rho = 0.9)
            : base(learningrate)
        {
            Rho = rho;
        }

        /// <summary>
        ///     动量因子
        /// </summary>
        public double Rho { set; get; }

        /// <summary>
        ///     负梯度的加权移动平均 => 参数更新方向
        /// </summary>
        public NDarray DeltaTheda { protected set; get; }

        internal override NDarray call(NDarray weight, NDarray grad, int epoch)
        {
            if (epoch == 0)
                DeltaTheda = np.zeros_like(weight);

            DeltaTheda = Rho * DeltaTheda - WorkLearningRate * grad;

            return weight + DeltaTheda;
        }
    }
}