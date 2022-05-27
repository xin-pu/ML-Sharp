using System;

namespace ML.Core.Optimizer
{
    public class Cosine : Annealing
    {
        /// <summary>
        ///     余弦衰减
        /// </summary>
        /// <param name="learningrate">初始学习率</param>
        /// <param name="totalepoch">迭代次数</param>
        public Cosine(double learningrate, int totalepoch) : base(learningrate)
        {
            TotalEpoch = totalepoch;
        }

        public int TotalEpoch { protected get; set; }

        internal override double UpdateLearningRate(int epoch)
        {
            return 0.5 * InitLearningRate * (1 + Math.Cos(epoch * Math.PI / TotalEpoch));
        }
    }
}