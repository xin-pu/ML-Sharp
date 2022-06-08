using System;

namespace ML.Core.Optimizers
{
    public class Cosine : Annealing
    {
        private int totalEpoch = 100;


        /// <summary>
        ///     余弦衰减
        /// </summary>
        public Cosine()
        {
        }

        /// <summary>
        ///     余弦衰减
        /// </summary>
        /// <param name="learningrate">初始学习率</param>
        public Cosine(double learningrate) : base(learningrate)
        {
        }

        /// <summary>
        ///     总迭代次数
        /// </summary>
        public int TotalEpoch
        {
            set => Set(ref totalEpoch, value);
            get => totalEpoch;
        }

        internal override double UpdateLearningRate(int epoch)
        {
            return 0.5 * InitLearningRate * (1 + Math.Cos(epoch * Math.PI / TotalEpoch));
        }
    }
}