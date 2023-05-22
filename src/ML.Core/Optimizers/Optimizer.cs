using System.ComponentModel;
using CommunityToolkit.Mvvm.ComponentModel;
using Numpy;

namespace ML.Core.Optimizers
{
    public abstract class Optimizer : ObservableObject, IDisposable
    {
        internal const double epsilon = 1E-7;

        private double _initLearningRate;
        private string _name = string.Empty;
        private double _workLearningRate;

        public Action<string>? AppendRecord;
        internal Func<NDarray, NDarray>? CalGradient;


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

        protected Optimizer()
        {
            Name = GetType().Name;
            InitLearningRate = WorkLearningRate = 1E-3;
        }

        [Category("Tag")]
        public string Name
        {
            protected set => SetProperty(ref _name, value);
            get => _name;
        }


        [Category("State")]
        public double WorkLearningRate
        {
            set => SetProperty(ref _workLearningRate, value);
            get => _workLearningRate;
        }

        [Category("Configuration")]
        public double InitLearningRate
        {
            set
            {
                SetProperty(ref _initLearningRate, value);
                WorkLearningRate = value;
            }
            get => _initLearningRate;
        }

        public abstract void Dispose();

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