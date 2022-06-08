using System;
using MvvmCross.ViewModels;
using Numpy;

namespace ML.Core.Optimizers
{
    public abstract class Optimizer : MvxViewModel
    {
        internal const double epsilon = 1E-7;

        private double _initLearningRate;
        private string _name;
        private double _workLearningRate;

        public Action<string> AppendRecord;

        internal Func<NDarray, NDarray> CalGradient;


        /// <summary>
        ///     优化器
        /// </summary>
        /// <param name="learningrate"></param>
        protected Optimizer(
            double learningrate)
        {
            InitLearningRate = WorkLearningRate = learningrate;
        }

        protected Optimizer()
        {
            Name = GetType().Name;
            InitLearningRate = WorkLearningRate = 1E-3;
        }

        public string Name
        {
            protected set => SetProperty(ref _name, value);
            get => _name;
        }

        public double WorkLearningRate
        {
            protected set => SetProperty(ref _workLearningRate, value);
            get => _workLearningRate;
        }

        public double InitLearningRate
        {
            protected set => SetProperty(ref _initLearningRate, value);
            get => _initLearningRate;
        }

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