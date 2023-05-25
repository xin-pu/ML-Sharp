using System.ComponentModel;
using AutoDiff;
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
        public abstract NDarray Call(NDarray weight, NDarray gradient, int epoch);


        /// <summary>
        ///     Core optimize function to update weights
        /// </summary>
        /// <param name="weight">Current Weights</param>
        /// <param name="calGradient">Function to calculation gradient</param>
        /// <param name="epoch"></param>
        /// <returns></returns>
        public virtual NDarray Call(NDarray weight, Variable[] variables, Term lossTerm, int epoch)
        {
            var gradientArray = lossTerm.Differentiate(variables, weight.GetData<double>());
            var g = np.array(gradientArray);
            return Call(weight, g, epoch);
        }
    }
}