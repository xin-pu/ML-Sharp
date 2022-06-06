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
        /// </summary>
        public NDarray DeltaTheda { protected set; get; }

        internal override NDarray call(NDarray weight, int epoch)
        {
            if (epoch == 0)
                DeltaTheda = np.zeros_like(weight);

            var next = Rho * DeltaTheda + DeltaTheda;
            var predWeight = weight + next;
            var grad = CalGradient(predWeight);

            DeltaTheda = Rho * DeltaTheda - WorkLearningRate * grad;
            return weight + DeltaTheda;
        }
    }
}