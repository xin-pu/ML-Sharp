using NumSharp;

namespace ML.Core.Optimizer
{
    public abstract class Annealing : SGD
    {
        /// <summary>
        ///     学习率退火
        /// </summary>
        /// <param name="learningrate"></param>
        protected Annealing(double learningrate)
            : base(learningrate)
        {
        }

        internal override NDArray call(NDArray weight, NDArray grad, int epoch)
        {
            WorkLearningRate = UpdateLearningRate(epoch);
            return base.call(weight, grad, epoch);
        }

        internal abstract double UpdateLearningRate(int epoch);
    }
}