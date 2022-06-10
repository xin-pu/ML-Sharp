using Numpy;

namespace ML.Core.Optimizers
{
    public abstract class Annealing : SGD
    {
        /// <summary>
        ///     学习率退火
        /// </summary>
        protected Annealing()
        {
        }

        /// <summary>
        ///     学习率退火
        /// </summary>
        /// <param name="learningrate"></param>
        protected Annealing(double learningrate)
            : base(learningrate)
        {
        }

        internal override NDarray call(NDarray weight, int epoch)
        {
            WorkLearningRate = UpdateLearningRate(epoch);
            var grad = CalGradient(weight);
            var delta = -grad * WorkLearningRate;
            return weight + delta;
        }

        internal abstract double UpdateLearningRate(int epoch);
    }
}