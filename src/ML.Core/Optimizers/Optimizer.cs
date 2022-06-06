using System;
using Numpy;

namespace ML.Core.Optimizers
{
    public abstract class Optimizer
    {
        internal const double epsilon = 1E-7;

        public Action<string> AppendRecord;

        internal Func<NDarray, NDarray> CalGradient;

        /// <summary>
        ///     优化器
        /// </summary>
        /// <param name="learningrate"></param>
        protected Optimizer(
            double learningrate)
        {
            Name = GetType().Name;
            InitLearningRate = WorkLearningRate = learningrate;
        }

        public string Name { protected set; get; }
        public double WorkLearningRate { protected set; get; }
        public double InitLearningRate { protected set; get; }

        /// <summary>
        ///     Core optimize function to update weights
        /// </summary>
        /// <param name="weight">Current Weights</param>
        /// <param name="calGradient">Function to calculation gradient</param>
        /// <param name="epoch"></param>
        /// <returns></returns>
        public NDarray Call(NDarray weight, Func<NDarray, NDarray> calGradient, int epoch)
        {
            CalGradient = calGradient;
            return call(weight, epoch);
        }

        internal abstract NDarray call(NDarray weight, int epoch);
    }
}