namespace ML.Core.Optimizers
{
    public class InverseTime : Annealing
    {
        private double _beta = 0.1;


        /// <summary>
        ///     逆时衰减
        /// </summary>
        public InverseTime()
        {
        }

        /// <summary>
        ///     逆时衰减
        /// </summary>
        /// <param name="learningrate">初始学习率</param>
        /// <param name="beta">衰减率</param>
        public InverseTime(double learningrate)
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
            return InitLearningRate / (1 + Beta * epoch);
        }
    }
}