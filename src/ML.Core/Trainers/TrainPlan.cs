using System;
using MvvmCross.ViewModels;

namespace ML.Core.Trainers
{
    public class TrainPlan : MvxViewModel
    {
        private int _batchSize;
        private int _epoch;

        public Func<bool> EarlyStoping { set; get; } = () => false;

        public int Epoch
        {
            get => _epoch;
            set => SetProperty(ref _epoch, value);
        }

        public int BatchSize
        {
            get => _batchSize;
            set => SetProperty(ref _batchSize, value);
        }

        public void Check()
        {
        }

        public void GiveEarlyStoping(Func<bool> func)
        {
            EarlyStoping = func;
        }
    }
}