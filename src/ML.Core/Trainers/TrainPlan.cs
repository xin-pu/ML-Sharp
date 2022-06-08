using System;
using GalaSoft.MvvmLight;

namespace ML.Core.Trainers
{
    public class TrainPlan : ViewModelBase
    {
        private int _batchSize;
        private int _epoch;

        public Func<bool> EarlyStoping { set; get; } = () => false;

        public int Epoch
        {
            get => _epoch;
            set => Set(ref _epoch, value);
        }

        public int BatchSize
        {
            get => _batchSize;
            set => Set(ref _batchSize, value);
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