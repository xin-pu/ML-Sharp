using System;
using Numpy;

namespace ML.Core.Optimizers
{
    public abstract class Optimizer
    {
        internal const double epsilon = 1E-7;

        public Action<string> AppendRecord;
        public Func<NDarray, NDarray, NDarray, (NDarray, NDarray)> calLoss;

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


        public NDarray Call(NDarray weight, NDarray grad, int epoch)
        {
            var gradientNDarray = np.reshape(grad, weight.shape);
            return call(weight, gradientNDarray, epoch);
        }

        internal abstract NDarray call(NDarray weight, NDarray grad, int epoch);
    }
}