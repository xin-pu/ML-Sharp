using Numpy;

namespace ML.Core.Optimizers
{
    public class Nesterov : Optimizer
    {
        /// <summary>
        ///     SGD with Nesterov Accelerated Gradient
        ///     Nesterov 加速梯度
        /// </summary>
        /// <param name="learningrate"></param>
        public Nesterov(double learningrate, double rho = 0.9)
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
        ///     θ=θ(t-1)+ρ*σθ(t-1)
        ///     σθ=ρ*σθ(t-1)-α*g(θ)
        ///     w=w+σθ
        ///     《神经网络与深度学习》 P167
        /// </summary>
        public NDarray DeltaWeight { protected set; get; }

        internal override NDarray call(NDarray weight, int epoch)
        {
            if (epoch == 0)
                DeltaWeight = np.zeros_like(weight);

            var theda = weight + Rho * DeltaWeight;
            var grad = CalGradient(theda);

            DeltaWeight = Rho * DeltaWeight - WorkLearningRate * grad;
            return weight + DeltaWeight;
        }
    }
}