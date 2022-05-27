using NumSharp;

namespace ML.Core.Optimizer
{
    public abstract class Optimizer
    {
        internal const double epsilon = 1E-7;

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


        public NDArray Call(NDArray weight, NDArray grad, int epoch)
        {
            return call(weight, grad, epoch);
        }

        internal abstract NDArray call(NDArray weight, NDArray grad, int epoch);
    }
}