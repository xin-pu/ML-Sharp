namespace ML.Core.Optimizers
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


        internal abstract double UpdateLearningRate(int epoch);
    }
}