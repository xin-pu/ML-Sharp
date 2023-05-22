namespace ML.Core.Optimizers
{
    public class Exponential : Annealing
    {
        private double _beta = 0.96;


        /// <summary>
        ///     学习率指数衰减
        /// </summary>
        /// <param name="learningrate">初始学习率</param>
        public Exponential()
        {
        }

        /// <summary>
        ///     学习率指数衰减
        /// </summary>
        /// <param name="learningrate">初始学习率</param>
        public Exponential(double learningrate)
            : base(learningrate)
        {
        }

        public double Beta
        {
            set => SetProperty(ref _beta, value);
            get => _beta;
        }

        internal override double UpdateLearningRate(int epoch)
        {
            return InitLearningRate * Math.Pow(Beta, epoch);
        }
    }
}