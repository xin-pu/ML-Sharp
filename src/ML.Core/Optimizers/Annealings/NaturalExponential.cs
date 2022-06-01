using System;

namespace ML.Core.Optimizers
{
    public class NaturalExponential : Annealing
    {
        /// <summary>
        ///     学习率自然指数衰减
        /// </summary>
        /// <param name="learningrate">初始学习率</param>
        /// <param name="beta">衰减率</param>
        public NaturalExponential(double learningrate, double beta = 0.04)
            : base(learningrate)
        {
            Beta = beta;
        }

        public double Beta { protected set; get; }

        internal override double UpdateLearningRate(int epoch)
        {
            return InitLearningRate * Math.Pow(Math.E, -Beta * epoch);
        }
    }
}