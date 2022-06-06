using Numpy;

namespace ML.Core.Optimizers
{
    public class Momentum : Optimizer
    {
        /// <summary>
        ///     SGD with Momentum
        ///     动量法
        ///     1.可以使用较大的学习率
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
        ///     σθ=ρσ(t-1)-α*gt
        ///     w=w+σθ
        ///     《神经网络与深度学习》 P167
        /// </summary>
        public NDarray DeltaTheda { protected set; get; }

        internal override NDarray call(NDarray weight, int epoch)
        {
            if (epoch == 0)
                DeltaTheda = np.zeros_like(weight);

            var grad = CalGradient(weight);
            DeltaTheda = Rho * DeltaTheda - WorkLearningRate * grad;

            return weight + DeltaTheda;
        }
    }
}