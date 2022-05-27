using System;

namespace ML.Core.Optimizer
{
    public class Exponential : Annealing
    {
        /// <summary>
        ///     学习率指数衰减
        /// </summary>
        /// <param name="learningrate">初始学习率</param>
        /// <param name="beta">衰减率</param>
        public Exponential(double learningrate, double beta = 0.96)
            : base(learningrate)
        {
            Beta = beta;
        }

        public double Beta { protected set; get; }

        internal override double UpdateLearningRate(int epoch)
        {
            return InitLearningRate * Math.Pow(Beta, epoch);
        }
    }
}