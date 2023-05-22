namespace ML.Core.Optimizers
{
    public class NaturalExponential : Annealing
    {
        private double _beta = 0.04;

        /// <summary>
        ///     学习率自然指数衰减
        /// </summary>
        public NaturalExponential()
        {
        }

        /// <summary>
        ///     学习率自然指数衰减
        /// </summary>
        /// <param name="learningrate">初始学习率</param>
        /// <param name="beta">衰减率</param>
        public NaturalExponential(double learningrate)
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
            return InitLearningRate * Math.Pow(Math.E, -Beta * epoch);
        }
    }
}