namespace ML.Core.Optimizer
{
    public class InverseTime : Annealing
    {
        /// <summary>
        ///     逆时衰减
        /// </summary>
        /// <param name="learningrate">初始学习率</param>
        /// <param name="beta">衰减率</param>
        public InverseTime(double learningrate, double beta = 0.1)
            : base(learningrate)
        {
            Beta = beta;
        }

        public double Beta { protected set; get; }

        internal override double UpdateLearningRate(int epoch)
        {
            return InitLearningRate / (1 + Beta * epoch);
        }
    }
}