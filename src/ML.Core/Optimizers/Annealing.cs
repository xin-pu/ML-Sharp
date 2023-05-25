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

        public override NDarray Call(NDarray weight, NDarray gradient, int epoch)
        {
            WorkLearningRate = UpdateLearningRate(epoch);
            var delta = -gradient * WorkLearningRate;
            return weight + delta;
        }

        internal abstract double UpdateLearningRate(int epoch);
    }
}